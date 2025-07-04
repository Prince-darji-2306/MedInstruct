let synth = window.speechSynthesis;
let currentUtter = null;

function startListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Your browser does not support speech recognition.");
        return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';

    const listeningModal = new bootstrap.Modal(document.getElementById('listeningModal'));
    listeningModal.show();

    recognition.start();

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('queryInput').value = transcript;
        listeningModal.hide();
        // set sessionStorage so we know it was from speech
        sessionStorage.setItem("fromVoice", "1");
        document.getElementById('searchButton').click();
    };

    recognition.onerror = function(event) {
        console.error("Speech recognition error:", event.error);
        listeningModal.hide();
    };

    recognition.onend = function() {
        listeningModal.hide();
    };
}

document.getElementById("searchButton").addEventListener("click", function(e) {
    const input = document.getElementById("queryInput").value.trim();
    if (input === "") {
        e.preventDefault();
        const emptyModal = new bootstrap.Modal(document.getElementById('emptyInputModal'));
        emptyModal.show();
    }
});

document.getElementById('micButton').addEventListener('click', startListening);

function speakResponse() {
    const text = document.getElementById("speakInstructions")?.innerText;
    if (!text) return;
    if (synth.speaking) {
        // stop speaking if already speaking
        synth.cancel();
        document.getElementById("replaySpeech").innerText = "🔊 Replay";
        return;
    }
    currentUtter = new SpeechSynthesisUtterance(text);
    currentUtter.rate = 0.9;
    synth.speak(currentUtter);
    document.getElementById("replaySpeech").innerText = "⏹ Stop";
    currentUtter.onend = function() {
        document.getElementById("replaySpeech").innerText = "🔊 Replay";
    }
}

document.getElementById('replaySpeech')?.addEventListener('click', speakResponse);

window.addEventListener('load', function() {
    if (sessionStorage.getItem("fromVoice") === "1") {
        // auto speak after results come
        speakResponse();
        sessionStorage.removeItem("fromVoice");
    }
});