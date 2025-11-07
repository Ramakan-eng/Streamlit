# app.py - Dedup / processing-guard fixes added
import os, io, json, time, queue, tempfile, threading, traceback
from datetime import datetime

import numpy as np
from scipy.io.wavfile import write as wav_write
from scipy.signal import butter, filtfilt
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

# ---------------- CONFIG ----------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing OPENAI_API_KEY in environment or .env")
    st.stop()

client = OpenAI(api_key=API_KEY)

SAMPLE_RATE = 16000
DEFAULT_GPT = "gpt-4o-mini"
DEFAULT_STT = "whisper-1"
DEFAULT_TTS = "gpt-4o-mini-tts"
EXIT_PHRASES = r"\b(stop|exit|quit|bye|goodbye|thank you|ok thanks)\b"

GLOBAL_QUEUE = queue.Queue()

# ---------------- HELPERS ----------------
def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def to_wav_bytes(arr: np.ndarray, sr=SAMPLE_RATE) -> bytes:
    arr = np.clip(arr, -1.0, 1.0).astype(np.float32)
    int16 = (arr * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav_write(buf, sr, int16)
    return buf.getvalue()

def save_bytes_to_file(b: bytes, suffix=".wav"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(b)
    return path

def highpass_filter(arr: np.ndarray, sr: int, cutoff_hz=80.0, order=4):
    ny = 0.5 * sr
    normal_cutoff = cutoff_hz / ny
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    try:
        return filtfilt(b, a, arr)
    except Exception:
        return arr

def detect_lang_safe(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"
    except Exception:
        return "en"

def sanitize_messages_for_worker(messages_list):
    clean = []
    for m in messages_list:
        if isinstance(m, dict) and "role" in m and "content" in m:
            clean.append({"role": str(m["role"]), "content": str(m["content"])})
        else:
            clean.append({"role": "assistant", "content": str(m)})
    return clean

# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(page_title="🎧  Voice Assistant", layout="wide")
st.title("🎧  Voice Assistant")

# initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful AI voice assistant."}]
if "chat" not in st.session_state:
    st.session_state.chat = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    gpt_model = st.selectbox("GPT Model", [DEFAULT_GPT, "gpt-4o"], index=0)
    stt_model = st.selectbox("STT Model", [DEFAULT_STT], index=0)
    tts_model = st.selectbox("TTS Model", [DEFAULT_TTS], index=0)
    voice_choice = st.selectbox("TTS Voice", ["alloy", "coral", "verse", "sage"], index=1)
    st.divider()
    st.subheader("Noise gate / filters")
    use_highpass = st.checkbox("Enable high-pass filter (remove low rumble)", value=True)
    hp_cutoff = st.slider("High-pass cutoff (Hz)", 40, 400, 80, step=10)
    rms_threshold = st.slider("RMS noise gate threshold (higher = stricter)", 0.002, 0.05, 0.01, 0.001)
    min_speech_ms = st.slider("Min speech length (ms)", 150, 800, 250, 50)
    st.divider()
    st.subheader("Assistant speech")
    speak_replies = st.checkbox("Speak assistant replies (TTS)", value=False)
    autoplay_tts = st.checkbox("Autoplay assistant audio in browser", value=True)
    st.divider()
    if st.button("Export chat (JSON)"):
        st.download_button("Download JSON", json.dumps(st.session_state.chat, indent=2), "chat.json", "application/json")

# ---------------- CHAT UI ----------------
chat_container = st.container()

def render_chat():
    with chat_container:
        for item in st.session_state.chat[-80:]:
            role = item["role"]
            text = item["text"]
            time_str = item.get("time","")
            if role == "user":
                st.markdown(
                    f"<div style='text-align:right; margin:6px 0'><small>{time_str}</small>"
                    f"<div style='display:inline-block;background:#0a74ff;color:#fff;padding:10px;border-radius:12px;max-width:78%'>{text}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:left; margin:6px 0'><small>{time_str}</small>"
                    f"<div style='display:inline-block;background:#111;color:#fff;padding:10px;border-radius:12px;max-width:78%'>{text}</div></div>",
                    unsafe_allow_html=True,
                )
            if item.get("audio"):
                st.audio(item["audio"])

render_chat()
st.divider()

# ---------------- INPUT LAYOUT ----------------
col_rec, col_type = st.columns([1.1, 2])
with col_rec:
    st.caption("Record (push-to-record) — browser mic")
    audio_dict = mic_recorder(start_prompt="Record", stop_prompt="Stop", just_once=True, use_container_width=True, format="wav", key="recorder_ui")
with col_type:
    typed_text = st.text_input("Or type your message", key="typed_input")
    # Send button added below using callback
    pass

# ---------------- WORKER ----------------
def pipeline_worker(prompt_text: str | None = None, audio_wav_path: str | None = None, snapshot_messages=None, speak_replies_flag=False):
    try:
        user_text = prompt_text
        if audio_wav_path:
            try:
                with open(audio_wav_path, "rb") as f:
                    stt_resp = client.audio.transcriptions.create(model=stt_model, file=f)
                user_text = (stt_resp.text or "").strip()
            except Exception as e:
                GLOBAL_QUEUE.put(("error", f"STT error: {e}", None))
                return
            GLOBAL_QUEUE.put(("user_transcript", user_text, audio_wav_path))

        if not user_text:
            GLOBAL_QUEUE.put(("error", "[No speech detected or empty input]", None))
            return

        import re
        if re.search(EXIT_PHRASES, user_text, flags=re.I):
            GLOBAL_QUEUE.put(("assistant", "Okay, goodbye! 👋", None))
            return

        messages_for_call = snapshot_messages.get("messages", []) if snapshot_messages else []
        messages_for_call = list(messages_for_call) + [{"role": "user", "content": user_text}]

        try:
            resp = client.chat.completions.create(model=gpt_model, messages=messages_for_call)
            reply = resp.choices[0].message.content.strip()
        except Exception as e:
            GLOBAL_QUEUE.put(("error", f"GPT error: {e}", None))
            return

        mp3_path = None
        if speak_replies_flag:
            try:
                speech = client.audio.speech.create(model=tts_model, voice=voice_choice, input=reply)
                fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                if hasattr(speech, "with_streaming_response"):
                    speech.with_streaming_response().stream_to_file(mp3_path)
                else:
                    speech.stream_to_file(mp3_path)
            except Exception as e:
                GLOBAL_QUEUE.put(("error", f"TTS error: {e}", None))
                mp3_path = None

        # Push assistant reply to queue once
        GLOBAL_QUEUE.put(("assistant", reply, mp3_path))

    except Exception as e:
        tb = traceback.format_exc()
        GLOBAL_QUEUE.put(("error", f"Worker exception: {e}\n{tb}", None))

# ---------------- MAIN-FACING ACTIONS ----------------
def append_user_and_start_worker_text(text: str):
    if not text: return

    # Append user once (main thread)
    st.session_state.chat.append({"role":"user","text":text,"time":now_str(),"audio":None})
    st.session_state.messages.append({"role":"user","content":text})

    # Prepare snapshot for worker (exclude the just-added user)
    snapshot = {"messages": sanitize_messages_for_worker(list(st.session_state.messages)[:-1])}

    # We already set processing=True at the top of callback; keep it true until done
    thread = threading.Thread(target=pipeline_worker, kwargs={
        "prompt_text": text, "snapshot_messages": snapshot, "speak_replies_flag": speak_replies
    }, daemon=True)
    thread.start()

    # Poll queue while running
    timeout = 40.0
    start_t = time.time()
    while True:
        while not GLOBAL_QUEUE.empty():
            kind, payload, audio_path = GLOBAL_QUEUE.get()
            if kind == "user_transcript":
                for i in range(len(st.session_state.chat)-1, -1, -1):
                    if st.session_state.chat[i]["role"] == "user":
                        st.session_state.chat[i]["text"] = payload or st.session_state.chat[i]["text"]
                        break
                chat_container.empty(); render_chat()
                continue

            if kind == "assistant":
                # DEDUP: skip if last assistant message is identical
                last_assistant_text = None
                for i in range(len(st.session_state.chat)-1, -1, -1):
                    if st.session_state.chat[i]["role"] == "assistant":
                        last_assistant_text = st.session_state.chat[i]["text"]
                        break
                if payload and payload == last_assistant_text:
                    # duplicate detected — ignore
                    st.session_state.processing = False
                    continue

                st.session_state.chat.append({"role":"assistant","text":payload,"time":now_str(),"audio":audio_path})
                st.session_state.messages.append({"role":"assistant","content":payload})
                st.session_state.processing = False
                chat_container.empty(); render_chat()
                if audio_path and autoplay_tts:
                    try:
                        st.audio(audio_path)
                    except Exception:
                        pass
                continue

            if kind == "error":
                # append error once
                last_assistant_text = None
                for i in range(len(st.session_state.chat)-1, -1, -1):
                    if st.session_state.chat[i]["role"] == "assistant":
                        last_assistant_text = st.session_state.chat[i]["text"]
                        break
                if payload and payload == last_assistant_text:
                    st.session_state.processing = False
                    continue
                st.session_state.chat.append({"role":"assistant","text":payload,"time":now_str(),"audio":None})
                st.session_state.processing = False
                chat_container.empty(); render_chat()
                continue

        if not thread.is_alive():
            break
        if time.time() - start_t > timeout:
            st.session_state.chat.append({"role":"assistant","text":"[Timeout waiting for response]","time":now_str(),"audio":None})
            st.session_state.processing = False
            chat_container.empty(); render_chat()
            break
        time.sleep(0.12)

# callback for Send button (safe to clear widget state)
def on_send_callback():
    # Prevent re-entrancy: if already processing, ignore repeated sends
    if st.session_state.get("processing", False):
        return
    text = st.session_state.get("typed_input","").strip()
    if not text:
        return
    # mark processing immediately so user can't start another request
    st.session_state["processing"] = True
    # clear input safely inside callback
    st.session_state["typed_input"] = ""
    append_user_and_start_worker_text(text)

# create Send button under the text input (ensures it's visually below input)
st.button("Send", on_click=on_send_callback)

# ---------------- AUDIO HANDLING (Stop -> placeholder -> transcription)
if audio_dict and "bytes" in audio_dict and audio_dict["bytes"] is not None and not st.session_state.processing:
    raw = audio_dict["bytes"]
    sr = int(audio_dict.get("sample_rate") or SAMPLE_RATE)

    # append placeholder that will be replaced by transcription
    st.session_state.chat.append({"role":"user","text":"[Recording… transcribing]","time":now_str(),"audio":None})
    st.session_state.messages.append({"role":"user","content":"[Recording audio]"})
    snapshot = {"messages": sanitize_messages_for_worker(list(st.session_state.messages)[:-1])}
    st.session_state.processing = True

    # handle raw bytes vs float array
    if isinstance(raw, (bytes, bytearray)):
        wav_path = save_bytes_to_file(raw, suffix=".wav")
        thread = threading.Thread(target=pipeline_worker, kwargs={
            "audio_wav_path": wav_path, "snapshot_messages": snapshot, "speak_replies_flag": speak_replies
        }, daemon=True)
        thread.start()
    else:
        arr = np.array(raw, dtype=np.float32).flatten()
        if use_highpass:
            arr = highpass_filter(arr, sr, cutoff_hz=hp_cutoff)
        rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
        if rms < rms_threshold or len(arr) < int(sr * min_speech_ms / 1000):
            # replace placeholder with message
            st.session_state.chat[-1]["text"] = "[Recording was too quiet or mostly noise — try again]"
            st.session_state.processing = False
            chat_container.empty(); render_chat()
        else:
            wav_bytes = to_wav_bytes(arr, sr=sr)
            wav_path = save_bytes_to_file(wav_bytes, suffix=".wav")
            thread = threading.Thread(target=pipeline_worker, kwargs={
                "audio_wav_path": wav_path, "snapshot_messages": snapshot, "speak_replies_flag": speak_replies
            }, daemon=True)
            thread.start()

    # poll queue while worker runs (so transcript & reply appear promptly)
    timeout = 60.0
    start_t = time.time()
    while True:
        while not GLOBAL_QUEUE.empty():
            kind, payload, audio_path = GLOBAL_QUEUE.get()
            if kind == "user_transcript":
                for i in range(len(st.session_state.chat)-1, -1, -1):
                    if st.session_state.chat[i]["role"] == "user":
                        st.session_state.chat[i]["text"] = payload or st.session_state.chat[i]["text"]
                        break
                chat_container.empty(); render_chat()
                continue
            if kind == "assistant":
                # DEDUP assistant same as above
                last_assistant_text = None
                for i in range(len(st.session_state.chat)-1, -1, -1):
                    if st.session_state.chat[i]["role"] == "assistant":
                        last_assistant_text = st.session_state.chat[i]["text"]
                        break
                if payload and payload == last_assistant_text:
                    st.session_state.processing = False
                    continue

                st.session_state.chat.append({"role":"assistant","text":payload,"time":now_str(),"audio":audio_path})
                st.session_state.messages.append({"role":"assistant","content":payload})
                st.session_state.processing = False
                chat_container.empty(); render_chat()
                if audio_path and autoplay_tts:
                    try:
                        st.audio(audio_path)
                    except Exception:
                        pass
                continue
            if kind == "error":
                last_assistant_text = None
                for i in range(len(st.session_state.chat)-1, -1, -1):
                    if st.session_state.chat[i]["role"] == "assistant":
                        last_assistant_text = st.session_state.chat[i]["text"]
                        break
                if payload and payload == last_assistant_text:
                    st.session_state.processing = False
                    continue

                st.session_state.chat.append({"role":"assistant","text":payload,"time":now_str(),"audio":None})
                st.session_state.processing = False
                chat_container.empty(); render_chat()
                continue
        # small safety checks
        if not any(t.is_alive() for t in threading.enumerate() if t.name != "MainThread"):
            break
        if time.time() - start_t > timeout:
            st.session_state.chat.append({"role":"assistant","text":"[Timeout waiting for audio response]","time":now_str(),"audio":None})
            st.session_state.processing = False
            chat_container.empty(); render_chat()
            break
        time.sleep(0.12)

# final render
chat_container.empty()
render_chat()
