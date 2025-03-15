const userInput = document.getElementById('user-input');
const chatWindow = document.getElementById('chat-window');
const micButton = document.getElementById('mic-button');
const sendButton = document.getElementById('send-button');
const spinner = document.getElementById('spinner');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Toggle recording on mic button click
micButton.addEventListener('click', () => {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});

// Start recording audio using MediaRecorder
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = event => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    mediaRecorder.onstop = handleRecordingStop;
    mediaRecorder.start();
    isRecording = true;
    // Change mic button icon to stop
    micButton.innerHTML = '<i class="fa fa-stop"></i>';
  } catch (err) {
    console.error("Error accessing microphone:", err);
  }
}

// Stop recording audio
function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
  }
}

// When recording stops, send audio to backend
function handleRecordingStop() {
  // Revert mic button icon back to mic
  micButton.innerHTML = '<i class="fa fa-microphone"></i>';
  const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
  showSpinner();

  // Create FormData for the audio file
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.wav');

  fetch('http://localhost:8000/audio', {
    method: 'POST',
    body: formData
  })
    .then(response => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    })
    .then(data => {
      hideSpinner();
      // Append transcribed text as user's message (right side)
      appendMessage("user", data.transcription);
      // Append LLM reply as bot's message (left side)
      appendMessage("bot", data.reply);
    })
    .catch(err => {
      hideSpinner();
      console.error("Error sending audio:", err);
      appendMessage("bot", "Sorry, an error occurred.");
    });
}

// Text message sending functionality
sendButton.addEventListener('click', sendTextMessage);
userInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') sendTextMessage();
});

function sendTextMessage() {
  const message = userInput.value.trim();
  if (message === "") return;
  appendMessage("user", message);
  userInput.value = "";
  showSpinner(); // Show spinner while waiting for the response
  fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: message })
  })
    .then(response => response.json())
    .then(data => {
      hideSpinner();
      appendMessage("bot", data.reply);
    })
    .catch(error => {
      hideSpinner();
      console.error('Error:', error);
      appendMessage("bot", "Sorry, an error occurred.");
    });
}

function appendMessage(sender, message) {
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message ' + sender;
  messageDiv.textContent = message;
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showSpinner() {
  spinner.style.display = 'block';
}

function hideSpinner() {
  spinner.style.display = 'none';
}
