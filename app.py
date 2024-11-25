import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from google.cloud import texttospeech
import tempfile
import speech_recognition as sr

# Set up API keys and environment variables
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_APPLICATION_CREDENTIALS_CONTENT = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_CONTENT"]

# Initialize components
tts_client = texttospeech.TextToSpeechClient()
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# Load document content for context
document_content = """
Hyundai Creta is a premium SUV with cutting-edge technology, exceptional performance, and an attractive design. 
It comes with features like a panoramic sunroof, ventilated seats, and multiple driving modes. Powered by a choice of petrol and diesel engines, the Creta offers a balance of performance and efficiency.
"""

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
def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Error: Speech recognition service is unavailable."

# Streamlit interface
st.title("Hyundai Creta Sales Audiobot")
st.write("Record your audio query to interact with the bot!")

# Audio Recording
recorded_audio = st.audio_input("Record your voice message")

# Handle Audio Input
if recorded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(recorded_audio.read())
        audio_file_path = temp_audio_file.name

    # Process audio
    with st.spinner("Processing your audio..."):
        user_query = speech_to_text(audio_file_path)

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
