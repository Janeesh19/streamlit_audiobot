import os
import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from google.cloud import texttospeech
import tempfile
import speech_recognition as sr


# Load environment variables from Streamlit secrets
GROQ_API_KEY = st.secrets["general"]["GROQ_API_KEY"]

# Write the Google service account JSON to a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_credentials_file:
    temp_credentials_file.write(st.secrets["general"]["GOOGLE_APPLICATION_CREDENTIALS_CONTENT"].encode())
    google_credentials_path = temp_credentials_file.name

# Initialize components
tts_client = texttospeech.TextToSpeechClient()
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# Load document content for context
with open("creta.txt", "r") as file:
    document_content = file.read()

# Define prompt template
combined_prompt = f"""
Act as an expert telephone sales agent for Hyundai, focusing on selling the Hyundai Creta. Engage with potential customers professionally and effectively. Base all responses on the provided context. Follow these guidelines:

Greeting: Start with a brief, friendly introduction.
Response Style:
    For regular questions: Provide crisp, concise answers. Aim for responses under 25 words.
    For technical questions: Offer more detailed explanations. Limit responses to 2-3 sentences for moderately technical queries, and up to 5 sentences for highly technical questions.

Key Principles:
    Listen actively to identify customer needs.
    Match Creta features to customer requirements.
    Highlight Creta's value proposition succinctly.
    Address objections briefly but effectively.
    Guide interested customers to next steps concisely.

Technical Knowledge: For engine specifications, performance metrics, or advanced features, provide accurate, detailed information. Use layman's terms to explain complex features unless the customer demonstrates technical expertise.

Tone: Maintain a friendly, professional tone. Adjust to the customer's communication style.
Uncertainty Handling: If unsure about a specific detail, briefly acknowledge the need to verify the information.

Always focus exclusively on the Hyundai Creta. Prioritize being helpful, honest, and customer-oriented.

Context:
{document_content}
Question: {{question}}
Helpful Answer:"""

PROMPT = PromptTemplate(input_variables=["question"], template=combined_prompt)
llm_chain = LLMChain(llm=llm, prompt=PROMPT)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to generate LLM response
def generate_response(prompt: str) -> str:
    response = llm_chain.run(question=prompt)
    return response

# Function for Text-to-Speech streaming
def text_to_speech_stream(text: str):
    chunks = text.split('. ')
    for chunk in chunks:
        if not chunk.strip():
            continue

        synthesis_input = texttospeech.SynthesisInput(text=chunk + '.')
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-IN",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        yield response.audio_content

# Function to save audio chunks to a temporary file for playback
def save_audio_to_file(audio_stream, suffix=".mp3"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio_file:
        for chunk in audio_stream:
            temp_audio_file.write(chunk)
        return temp_audio_file.name  # Return the file path

# Function for speech-to-text
def speech_to_text():
    recognizer = sr.Recognizer()
    fs = 44100  # Sampling frequency
    duration = 7  # Duration in seconds

    # Create a placeholder for the "Listening..." message
    message_placeholder = st.empty()
    message_placeholder.info("Listening... Speak now.")  # Display the message

    try:
        # Record audio from the microphone
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until the recording is finished

        # Clear the "Listening..." message
        message_placeholder.empty()

        # Save audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            wav.write(temp_audio_file.name, fs, audio_data)
            user_audio_path = temp_audio_file.name

        # Recognize speech from the saved audio file
        with sr.AudioFile(user_audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        return text

    except sr.UnknownValueError:
        message_placeholder.empty()
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        message_placeholder.empty()
        return "Error: Speech recognition service is unavailable."

# Streamlit interface
st.title("Hyundai Creta Sales Audiobot")
st.write("Ask your questions using your voice!")

if st.button("Start Recording"):
    # Record and process user query
    user_query = speech_to_text()
    if user_query:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Generate bot response
        with st.spinner("Generating response..."):
            response = generate_response(user_query)
            audio_stream = text_to_speech_stream(response)

            # Save bot response audio
            bot_audio_path = save_audio_to_file(audio_stream)
            st.session_state.chat_history.append({"role": "bot", "content": response, "audio": bot_audio_path})

# Display chat history with bot's audio
st.write("### Chat History")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Bot:** {message['content']}")
        if message.get("audio"):
            with open(message["audio"], "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
