const sourceSelect = document.getElementById("source-select");
const videoSource = document.getElementById("video-source");
const videoPlayer = document.getElementById("video-player");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

function loadVideo(source) {
  videoSource.src = `/video/${source}`;
  videoPlayer.load();
}

function addMessage(text, role, extraClass = "") {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = `bubble ${extraClass}`;
  bubble.textContent = text;
  msg.appendChild(bubble);
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
  return bubble;
}

sourceSelect.addEventListener("change", () => loadVideo(sourceSelect.value));

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  addMessage(question, "user");
  chatInput.value = "";
  const pendingBubble = addMessage("생각하는 중...", "assistant", "pending");
  const sendButton = chatForm.querySelector("button");
  sendButton.disabled = true;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: sourceSelect.value, question }),
    });
    const data = await res.json();
    if (res.ok) {
      pendingBubble.textContent = data.answer;
      pendingBubble.classList.remove("pending");
    } else {
      pendingBubble.textContent = data.error || "오류가 발생했습니다.";
      pendingBubble.classList.add("error");
      pendingBubble.classList.remove("pending");
    }
  } catch (err) {
    pendingBubble.textContent = `네트워크 오류: ${err}`;
    pendingBubble.classList.add("error");
    pendingBubble.classList.remove("pending");
  } finally {
    sendButton.disabled = false;
  }
});

if (sourceSelect.options.length > 0) {
  loadVideo(sourceSelect.value);
}
