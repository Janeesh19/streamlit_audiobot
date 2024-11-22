import streamlit as st
import openai
from gtts import gTTS
from tempfile import NamedTemporaryFile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, ClientSettings
import whisper
import queue
import numpy as np

# Set up OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Replace with your OpenAI API key

# Whisper Model for Speech-to-Text
model = whisper.load_model("base")  # Whisper for real-time transcription

# GPT-3.5 API Interaction
def ask_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an AI assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# Text-to-Speech Function
def text_to_speech(text):
    tts = gTTS(text, lang="en")
    with NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts.save(tmpfile.name)
        return tmpfile.name

# Audio Processor for Real-Time Transcription
class SpeechToTextProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = queue.Queue()
        self.result_text = ""

    def recv_audio(self, frame):
        audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
        self.audio_buffer.put(audio_data)
        return frame

    def transcribe_audio(self):
        if not self.audio_buffer.empty():
            audio_frames = []
            while not self.audio_buffer.empty():
                audio_frames.append(self.audio_buffer.get())
            
            audio_data = np.concatenate(audio_frames, axis=0)
            audio_path = NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(audio_path, "wb") as f:
                f.write(audio_data.tobytes())
            
            # Transcribe with Whisper
            result = model.transcribe(audio_path)
            self.result_text = result["text"]
            return self.result_text
        return ""

# Streamlit App
def main():
    st.title("Real-Time Audio Bot 🎙️")
    st.header("Talk to GPT-3.5 in Real-Time")

    # WebRTC for Audio Streaming
    processor = SpeechToTextProcessor()
    webrtc_ctx = webrtc_streamer(
        key="real-time-audio",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=lambda: processor,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        ),
    )

    if webrtc_ctx.state.playing:
        st.write("🎤 Listening...")
        
        # Transcribe and Interact with GPT-3.5
        if st.button("Transcribe and Respond"):
            with st.spinner("Transcribing and generating response..."):
                user_input = processor.transcribe_audio()
                if user_input:
                    st.write(f"**You Said:** {user_input}")
                    
                    # Get GPT-3.5 response
                    response = ask_gpt(user_input)
                    st.write(f"**GPT-3.5 Says:** {response}")
                    
                    # Convert response to audio
                    audio_file = text_to_speech(response)
                    audio_bytes = open(audio_file, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3", start_time=0)
                else:
                    st.warning("No audio detected or transcription failed!")

if __name__ == "__main__":
    main()
