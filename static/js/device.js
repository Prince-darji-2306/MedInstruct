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
    } else {
        // show the spinner
        const spinnerModal = new bootstrap.Modal(document.getElementById('spinnerModal'));
        spinnerModal.show();
    }
});

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.getElementById('sidebarToggle');

    if (sidebar.classList.contains('visible')) {
        sidebar.classList.remove('visible');
        toggleBtn.style.transform = 'rotate(0deg)';
        toggleBtn.textContent = '>';
        
    } else {
        sidebar.classList.add('visible');
        toggleBtn.style.transform = 'rotate(180deg)';
    }
}


document.getElementById('sidebarToggle').addEventListener('click', toggleSidebar);

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

// Close sidebar on window resize if screen becomes larger
window.addEventListener('resize', function() {
    if (window.innerWidth > 768) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar.classList.contains('visible')) {
            sidebar.classList.remove('visible');
        }
    }
});

// Handle escape key to close sidebar
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && window.innerWidth <= 768) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar.classList.contains('visible')) {
            toggleSidebar();
        }
    }
});