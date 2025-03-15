# import streamlit as st
# from llm import get_response  # Import the function
#
# st.title("Medical ChatBot")
#
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# # Accept user input
# if prompt := st.chat_input("Ask me anything about health..."):
#     with st.chat_message("user"):
#         st.markdown(prompt)
#
#     st.session_state.messages.append({"role": "user", "content": prompt})
#
#     # Get response from LLM
#     assistant_response = get_response(prompt)
#
#     with st.chat_message("assistant"):
#         st.markdown(assistant_response)
#
#     st.session_state.messages.append({"role": "assistant", "content": assistant_response})
#
#
#
#
import streamlit as st
import whisper
import numpy as np
import base64
import io
from llm import get_response

st.title("Medical ChatBot with Audio Support")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None


# Load Whisper Model
@st.cache_resource()
def load_whisper_model():
    return whisper.load_model("base")


model = load_whisper_model()

# JavaScript-based audio recorder
audio_recorder_js = """
    <script>
    let mediaRecorder;
    let audioChunks = [];

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    let audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    let reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        let base64data = reader.result.split(',')[1];
                        fetch('/upload_audio', {
                            method: 'POST',
                            body: JSON.stringify({ audio: base64data }),
                            headers: { 'Content-Type': 'application/json' }
                        }).then(response => response.json())
                          .then(data => {
                              window.parent.postMessage({ type: 'AUDIO_UPLOADED', data: data }, '*');
                          });
                    };
                };
            });
    }

    function stopRecording() {
        if (mediaRecorder) {
            mediaRecorder.stop();
        }
    }

    window.addEventListener("message", (event) => {
        if (event.data === "START_RECORDING") {
            startRecording();
        } else if (event.data === "STOP_RECORDING") {
            stopRecording();
        }
    });
    </script>
"""

st.markdown(audio_recorder_js, unsafe_allow_html=True)

# Buttons for controlling recording
if st.button("🎤 Start Recording"):
    st.session_state.audio_data = None
    st.markdown("<script>window.postMessage('START_RECORDING', '*');</script>", unsafe_allow_html=True)

if st.button("⏹ Stop Recording"):
    st.markdown("<script>window.postMessage('STOP_RECORDING', '*');</script>", unsafe_allow_html=True)

# Process audio when received
if st.session_state.audio_data:
    st.write("Recording complete! Transcribing...")
    audio_bytes = base64.b64decode(st.session_state.audio_data)

    filename = "temp_audio.wav"
    with open(filename, "wb") as f:
        f.write(audio_bytes)

    # Transcribe using Whisper
    transcription = model.transcribe(filename)
    transcribed_text = transcription["text"]

    st.write(f"**You said:** {transcribed_text}")

    # Get LLM response
    assistant_response = get_response(transcribed_text)

    # Display response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Store messages
    st.session_state.messages.append({"role": "user", "content": transcribed_text})
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Text input chat
if prompt := st.chat_input("Ask me anything about health..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    assistant_response = get_response(prompt)

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})



