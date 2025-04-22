import streamlit as st
from transformers import pipeline
import torch
import time
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Phi and Gemini imports
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Streamlit 
st.set_page_config(
    page_title="agent_AI_system",
    layout="centered"
)

st.title("Multimodal AI Agent")
st.header("Upload audio or video and receive automated insights")

@st.cache_resource
def load_audio_model():
    return pipeline(
        "automatic-speech-recognition",
        "openai/whisper-tiny",
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
        device=device
    )

audio_pipe = load_audio_model()

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Media Analyzer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

media_agent = initialize_agent()

# for formatting audio transcription
def format_transcription(transcription):
    formatted_text = ""
    for line in transcription['chunks']:
        text = line["text"]
        ts = line["timestamp"]
        formatted_text += f"[{ts[0]}:{ts[1]}] {text}\n"
    return formatted_text.strip()

# file selection
media_type = st.radio("Choose media type to upload", ["Audio", "Video"])

transcribed_text = None
# Initialize session state variables to store them till the session ends
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = None
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None

if media_type == "Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
    if audio_file:
        st.audio(audio_file)
        language = st.selectbox("Choose language of the audio", ["English", "Hindi", "French"])
        task = st.selectbox("Choose the task", ["transcribe", "translate"])

        if st.button("Analyze Audio"):
            with st.spinner("Processing audio..."):
                temp_dir = "temp_audio_dir"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, audio_file.name)

                with open(temp_path, "wb") as f:
                    f.write(audio_file.getbuffer())

                transcription = audio_pipe(temp_path, generate_kwargs={"language": language, "task": task})
                formatted = format_transcription(transcription)
                st.session_state.transcribed_text = formatted  # store in session state

                prompts = [
                    "1. Summarization: Provide a concise summary of the overall discussion or content.",
                    "2. Action Items and Decisions: Identify any clear action items, tasks, or decisions mentioned or implied.",
                    "3. Follow-Ups Required: Mention any suggested or implied follow-up actions or unanswered questions.",
                    "4. Confidence and Clarity Analysis: Evaluate the confidence level and clarity of the speaker(s)—highlight any parts where the intent was uncertain, vague, or speculative.",
                    "5. Domain Classification: Identify the primary domain or context of the conversation (e.g., technical, business, healthcare, education, etc.)."
                ]

                insights = ""
                for p in prompts:
                    full_prompt = f"Given the following audio transcription:\n\n{formatted}\n\n{p}"
                    response = media_agent.run(full_prompt)
                    insights += f"### {p}\n{response.content}\n\n"

                st.subheader("Automated Insights")
                st.markdown(insights)

                os.remove(temp_path)

        if st.session_state.transcribed_text:
            user_query = st.text_area("Enter your question about the audio")
            if st.button("Run Custom Query"):
                custom_prompt = f"Based on...\n\n{st.session_state.transcribed_text}\n\n{user_query}"
                response = media_agent.run(custom_prompt)
                st.subheader("AI Response to Custom Query")
                st.markdown(response.content)

elif media_type == "Video":
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path)

        if st.button("Analyze Video"):
            try:
                with st.spinner("Processing video and gathering insights..."):
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                        st.session_state.processed_video = processed_video  

                    prompts = [
                        "1. Summarization: Provide a concise summary of the overall discussion or content.",
                        "2. Action Items and Decisions: Identify any clear action items, tasks, or decisions mentioned or implied.",
                        "3. Follow-Ups Required: Mention any suggested or implied follow-up actions or unanswered questions.",
                        "4. Confidence and Clarity Analysis: Evaluate the confidence level and clarity of the speaker(s)—highlight any parts where the intent was uncertain, vague, or speculative.",
                        "5. Domain Classification: Identify the primary domain or context of the conversation (e.g., technical, business, healthcare, education, etc.)."
                    ]


                    insights = ""
                    for p in prompts:
                        video_prompt = f"Analyze the uploaded video for content and context.\n\n{p}"
                        response = media_agent.run(video_prompt, videos=[processed_video])
                        insights += f"### {p}\n{response.content}\n\n"

                    st.subheader("Automated Insights")
                    st.markdown(insights)

            except Exception as error:
                st.error(f"An error occurred: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)

        if st.session_state.processed_video:
            user_query = st.text_area("Enter your question about the video")
            if st.button("Run Custom Query on Video"):
                try:
                    custom_prompt = f"Analyze...\n\n{user_query}"
                    response = media_agent.run(custom_prompt, videos=[st.session_state.processed_video])
                    st.subheader("AI Response to Custom Query")
                    st.markdown(response.content)
                except Exception as error:
                    st.error(f"An error occurred: {error}")

st.markdown("""
<style>
    .stTextArea textarea {
        height: 100px;
    }
</style>
""", unsafe_allow_html=True)
