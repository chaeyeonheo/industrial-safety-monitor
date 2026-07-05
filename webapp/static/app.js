const sourceSelect = document.getElementById("source-select");
const videoSource = document.getElementById("video-source");
const videoPlayer = document.getElementById("video-player");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const sourceStatus = document.getElementById("source-status");
const statEvents = document.getElementById("stat-events");
const statFalls = document.getElementById("stat-falls");
const statPpe = document.getElementById("stat-ppe");
const statTracks = document.getElementById("stat-tracks");
const insightText = document.getElementById("insight-text");
const quickChips = document.querySelectorAll(".quick-chip");

let activeTimeline = null;
let activeObjectUrl = null;
let localMode = false;

function setIfPresent(element, value) {
  if (element) {
    element.textContent = value;
  }
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

function setStatus(text, kind = "") {
  sourceStatus.textContent = text;
  sourceStatus.dataset.kind = kind;
}

function resetTimelineUI() {
  setIfPresent(statEvents, "0");
  setIfPresent(statFalls, "0");
  setIfPresent(statPpe, "0");
  setIfPresent(statTracks, "0");
  setIfPresent(insightText, "데모 source를 선택하거나 파일을 드롭하면 요약이 표시됩니다.");
}

function summarizeTimeline(timeline) {
  const events = Array.isArray(timeline) ? timeline : [];
  const uniqueTracks = new Set();
  const eventTypes = new Map();
  let fallCount = 0;
  let ppeCount = 0;

  for (const event of events) {
    if (!event || typeof event !== "object") {
      continue;
    }
    if (event.track_id !== undefined && event.track_id !== null) {
      uniqueTracks.add(event.track_id);
    }
    const type = event.event_type || "unknown";
    eventTypes.set(type, (eventTypes.get(type) || 0) + 1);
    if (type === "fall_suspected") fallCount += 1;
    if (type === "ppe_missing") ppeCount += 1;
  }

  setIfPresent(statEvents, String(events.length));
  setIfPresent(statFalls, String(fallCount));
  setIfPresent(statPpe, String(ppeCount));
  setIfPresent(statTracks, String(uniqueTracks.size));

  if (!events.length) {
    setIfPresent(insightText, "타임라인에 기록된 이벤트가 없습니다.");
    return;
  }

  const topEvents = [...eventTypes.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2)
    .map(([name, count]) => `${name} ${count}건`)
    .join(", ");
  const start = events[0]?.timestamp_sec ?? 0;
  const end = events[events.length - 1]?.timestamp_sec ?? 0;
  setIfPresent(insightText, `이 타임라인은 ${start}초부터 ${end}초까지의 이벤트를 담고 있습니다. 주요 이벤트는 ${topEvents}입니다.`);
}

function loadVideo(source) {
  if (activeObjectUrl) {
    URL.revokeObjectURL(activeObjectUrl);
    activeObjectUrl = null;
  }
  localMode = false;
  activeTimeline = null;
  videoSource.src = `/video/${source}`;
  videoPlayer.load();
  setStatus(`데모 source: ${source}`, "ok");
  resetTimelineUI();
}

function isVideoFile(file) {
  return file.type.startsWith("video/") || /\.(mp4|webm|mov|m4v)$/i.test(file.name);
}

function isJsonFile(file) {
  return file.type === "application/json" || /\.json$/i.test(file.name);
}

async function loadLocalFiles(files) {
  const fileList = Array.from(files);
  const videoFile = fileList.find(isVideoFile);
  const timelineFile = fileList.find(isJsonFile);

  if (!videoFile && !timelineFile) {
    setStatus("mp4 영상 파일이나 event_timeline json 파일을 드래그하세요.", "error");
    return;
  }

  localMode = true;
  activeTimeline = null;
  resetTimelineUI();

  if (videoFile) {
    if (activeObjectUrl) {
      URL.revokeObjectURL(activeObjectUrl);
    }
    activeObjectUrl = URL.createObjectURL(videoFile);
    videoSource.src = activeObjectUrl;
    videoPlayer.load();
    setStatus(`로컬 비디오 로드됨: ${videoFile.name}`, "ok");
  }

  if (timelineFile) {
    try {
      const parsed = JSON.parse(await timelineFile.text());
      activeTimeline = Array.isArray(parsed) ? parsed : (parsed.timeline ?? null);
      if (!Array.isArray(activeTimeline)) {
        throw new Error("JSON 최상위에 이벤트 배열이 없습니다.");
      }
      summarizeTimeline(activeTimeline);
      setStatus(
        `${videoFile ? `로컬 비디오 ${videoFile.name}` : "현재 영상"} + 타임라인 ${timelineFile.name} 사용 중`,
        "ok"
      );
    } catch (err) {
      activeTimeline = null;
      setStatus(`타임라인 JSON을 읽을 수 없습니다: ${err}`, "error");
    }
  } else if (videoFile) {
    setStatus("비디오만 로드됨. 질문 응답을 하려면 matching event_timeline json도 드롭하세요.", "warn");
  }
}

function setDemoMode(source) {
  localMode = false;
  loadVideo(source);
}

sourceSelect.addEventListener("change", () => loadVideo(sourceSelect.value));

quickChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    chatInput.value = chip.dataset.question || "";
    chatInput.focus();
  });
});

if (dropZone && fileInput) {
  const preventDefaults = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, preventDefaults);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add("dragover"));
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove("dragover"));
  });

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => loadLocalFiles(fileInput.files));
  dropZone.addEventListener("drop", (event) => loadLocalFiles(event.dataTransfer.files));
}

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
    const payload = { question };
    if (activeTimeline) {
      payload.timeline = activeTimeline;
      payload.source = localMode ? "uploaded" : sourceSelect.value;
    } else if (!localMode && sourceSelect.value) {
      payload.source = sourceSelect.value;
    } else {
      throw new Error("먼저 데모 결과를 생성하거나 mp4와 event_timeline json을 함께 드래그하세요.");
    }

    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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
  setDemoMode(sourceSelect.value);
} else {
  setStatus("데모 결과가 없습니다. mp4와 event_timeline json을 드래그 앤 드롭하세요.", "warn");
  resetTimelineUI();
}
