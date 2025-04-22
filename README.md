# agentic_AI_system

An intelligent and interactive Streamlit-based web application that allows users to upload files, automatically **transcribe**, **analyze**, and **extract summaries, decisions, follow-ups**, and more using **Whisper (OpenAI)** and **gemini-2.0-flash-exp** from Google.

## Requirement of FFmpeg
FFmpeg is a free and open-source software project consisting of a suite of libraries and programs for handling video, audio, and other multimedia files.
Our app uses OpenAI's Whisper (transformers pipeline) for speech-to-text, which requires audio in specific formats (WAV/PCM).Also, When processing videos (MP4/MOV/AVI), FFmpeg extracts audio tracks from videos for transcription. Therefore, FFmpeg is important for handling these conversions.

### App Interface
![upload interface](interface.png)

## Features

- **Audio Transcription** using Whisper ASR (**required**)
- **Video Understanding** via Gemini model (**additional work done**)
- Detailed AI-generated insights:
  - Summary
  - Action items & decisions
  - Follow-up tasks
  - Confidence checks
  - Domain/topic detection
- User-customizable prompt after required analyis (**additional work done**) 
- Supported for `.mp3`, `.wav`, `.ogg`, `.mp4`, `.mov`, `.avi`
- Lightweight
  

## Audio2Text Conversion — Whisper by OpenAI

Whisper is a general-purpose speech recognition model trained on a large dataset of multilingual and multitask supervised data collected from the web. Whisper officially supports 99 languages for trancription of speech, and can translate other 98 languages to english.


### Whisper vs Other ASR Models

| Model           | Multilingual | Speed (tiny) | Accuracy | Timestamps | GPU Recommended |
|----------------|--------------|--------------|----------|------------|------------------|
| **Whisper**     | Yes       | Fast       | High   | Yes      | Optional      |
| Wav2Vec2 (HF)   | Limited   | Medium     | Good   | No       | Yes           |
| Google Speech API | Yes     | Fast       | Good   | No       | Cloud-only    |


## gemini-2.0-flash-exp Model

gemini-2.0-flash-exp is one of the most **advanced multimodal LLMs** that supports **video, text, and image** inputs. Google gemini is considered to be a better storyteller than other open-source models, making it a good fit for analyzing full-length videos or long transcriptions. Also, DuckDuckGo tool is used for doing real-time web search for giving supplement information, with main advantage being open-sourced.

### Gemini vs Other LLMs

| Model         | Multimodal | Context Length | Reasoning | Video Support | 
|---------------|------------|----------------|-----------|----------------|
| **gemini-2.0-flash** | Yes     | Long (1M tokens) | Strong | Yes         | 
| GPT-4 (OpenAI) | Partial | Medium        | Strong | No           | 
| Claude 3       | Yes     | Long          | Okay   | Limited      | 



### File Upload (Audio/Video)
![upload interface](images/upload-interface.png)

### AI-Generated Insights
![summary result](images/summary-insights.png)

### User Prompt for Deeper Analysis
![custom prompt](images/user-prompt.png)


## How It Works

1. **Upload File**: User selects Audio or Video and uploads a file.
2. **Initial Processing**:
   - Audio: Transcribed via Whisper
   - Video: Uploaded and pre-processed for Gemini
3. **Gemini Agent Response**:
   - Generates structured insights:
     - Summary
     - Action items / decisions
     - Follow-ups
     - Confidence level
     - Domain detection
4. **User Prompt (Optional)**: Allows further custom questioning after initial analysis.

### Future Work


1. **Persistent User History with Database Integration**
   - We can maintain logs of input media files and metadata, AI-generated responses (summaries, action items, decisions, etc.) and user questions and feedback.
   - These can be leveraged for generating cumulative insights and detect themes or trends over time.

2. **Analytics Dashboard**
   - Add a visual dashboard using Plotly/Streamlit charts to show frequency of topics discussed, sentimental trends and domain classification stats over time

Due to having a time constraint for this task, I will surely want to try out the future works.

Overall the Exercise was fun and kept me interested into it, till the submission.


### Load all the requirements

Before running the app locally, make sure you have **Python ≥ 3.9** installed. Then, follow the steps below to install all dependencies and run the app.

```bash
# Create a virtual environment 
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt

# On Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# On macOS using Homebrew
brew install ffmpeg

# On Windows
# Download from: https://ffmpeg.org/download.html


# Once everything is installed properly, we can launch the streamlit app
streamlit run main.py
