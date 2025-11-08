import os
import time
import threading
import queue
import re
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
import pygame
from dotenv import load_dotenv
import streamlit as st

# ---------------- CONFIG ----------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLE_RATE = 16000
AUDIO_FILE = "user.wav"
REPLY_FILE = "reply.mp3"

GPT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
STT_MODEL = "whisper-1"

EXIT_RE = re.compile(r"\b(stop|exit|quit|thank you|goodbye|bye|ok thanks)\b", re.I)

# ---------------- THREAD COMMUNICATION ----------------
events_q: queue.Queue[tuple[str, str]] = queue.Queue()
memory_global: list[dict] = []
is_listening = False

# ---------------- AUDIO UTILS ----------------
def record_audio(duration=4):
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    write(AUDIO_FILE, SAMPLE_RATE, (audio * 32767).astype(np.int16))

def transcribe_audio() -> str:
    try:
        with open(AUDIO_FILE, "rb") as f:
            out = client.audio.transcriptions.create(model=STT_MODEL, file=f, language="en")
        return (out.text or "").strip()
    except Exception as e:
        return f"[Whisper error: {e}]"

def ask_gpt(memory: list[dict]) -> str:
    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "system",
                       "content": "You are a friendly, concise voice assistant."}] + memory
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT error: {e}]"

def speak_openai(text: str):
    try:
        speech = client.audio.speech.create(model=TTS_MODEL, voice="coral", input=text)
        speech.stream_to_file(REPLY_FILE)
        pygame.mixer.init()
        pygame.mixer.music.load(REPLY_FILE)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.quit()
    except Exception as e:
        print(f"[TTS/Playback error] {e}")

# ---------------- BACKGROUND LISTEN LOOP ----------------
def listen_loop():
    global is_listening, memory_global
    while is_listening:
        record_audio(duration=4)
        user_text = transcribe_audio()
        if not user_text:
            continue

        events_q.put(("user", user_text))
        memory_global.append({"role": "user", "content": user_text})

        if EXIT_RE.search(user_text):
            reply = "Okay, goodbye! 👋"
            memory_global.append({"role": "assistant", "content": reply})
            events_q.put(("assistant", reply))
            speak_openai(reply)
            is_listening = False
            break

        reply = ask_gpt(memory_global)
        memory_global.append({"role": "assistant", "content": reply})
        events_q.put(("assistant", reply))
        speak_openai(reply)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="🎧 Real-Time Voice Assistant", page_icon="🎤", layout="wide")

# --- CSS ---
st.markdown(
    """
    <style>
    .stButton>button {
        background: linear-gradient(90deg,#0099ff,#00cc88);
        color:white;
        border:none;
        border-radius:10px;
        padding:0.6em 1.2em;
        font-weight:600;
    }
    .bubble-wrap {display:flex; flex-direction:column; gap:0.4rem;}
    .user-bubble {
        background-color:#004aad;
        color:white;
        border-radius:15px;
        padding:10px 14px;
        margin:2px 0;
        max-width:78%;
        align-self:flex-end;
        box-shadow: 0 2px 8px rgba(0,0,0,.25);
    }
    .bot-bubble {
        background-color:#262730;
        color:#fff;
        border-radius:15px;
        padding:10px 14px;
        margin:2px 0;
        max-width:78%;
        align-self:flex-start;
        box-shadow: 0 2px 8px rgba(0,0,0,.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎧 Real-Time Voice Assistant")
st.caption("Talk naturally — your speech and the AI’s reply appear below in real time 💬")

# --- Initialize Chat State ---
if "chat" not in st.session_state:
    st.session_state.chat = []
if "running" not in st.session_state:
    st.session_state.running = False

# --- Buttons ---
col1, col2, col3 = st.columns([1,1,3])
with col1:
    if not st.session_state.running:
        if st.button("🎙️ Start Talking", use_container_width=True):
            st.session_state.running = True
            globals()["is_listening"] = True
            threading.Thread(target=listen_loop, daemon=True).start()
    else:
        if st.button("🛑 Stop Talking", use_container_width=True):
            st.session_state.running = False
            globals()["is_listening"] = False

with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat.clear()
        memory_global.clear()

# --- Chat container ---
chat_box = st.empty()

# --- Live update loop ---
# This uses Streamlit's rerun-safe refresh mechanism.
if st.session_state.running:
    while st.session_state.running:
        # Drain events from background thread
        drained = False
        while True:
            try:
                role, text = events_q.get_nowait()
            except queue.Empty:
                break
            else:
                st.session_state.chat.append((role, text))
                drained = True

        # Render chat updates live
        with chat_box.container():
            st.markdown("<div class='bubble-wrap'>", unsafe_allow_html=True)
            for role, msg in st.session_state.chat[-30:]:
                if role == "user":
                    st.markdown(f"<div class='user-bubble'>🧍‍♂️ {msg}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot-bubble'>🤖 {msg}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Small delay to keep UI responsive
        time.sleep(0.8)

# --- Static render when not talking ---
else:
    with chat_box.container():
        st.markdown("<div class='bubble-wrap'>", unsafe_allow_html=True)
        for role, msg in st.session_state.chat[-30:]:
            if role == "user":
                st.markdown(f"<div class='user-bubble'>🧍‍♂️ {msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-bubble'>🤖 {msg}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
