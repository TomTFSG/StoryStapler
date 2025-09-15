let windowMargin = 10;

document.addEventListener("click", (e) => {
  if (e.target.classList.contains("edit-scene-btn")) {
    const wrapper = e.target.closest(".sceneDisplay").parentElement;
    const formId = wrapper.id.replace("-wrapper", "");
    const sceneId = wrapper.dataset.sceneId;
    const startSec = parseFloat(wrapper.dataset.start_time);
    const endSec = parseFloat(wrapper.dataset.end_time);

    // Rebuild scene object — you might want to save more data to wrapper.dataset if needed
    const scene = {
      scene_id: sceneId,
      title: window.currentSceneTitle,  // you may need a map if you queue multiple scenes
      segment: [secondsToTime(startSec), secondsToTime(endSec)],
      moviecode: extractMovieCodeFromSceneId(sceneId), // You'll define this helper
      total_scenes: 1,  // or look it up
      type: "movie",  // or save type in dataset
      desc: "",       // optionally preload description
      dialogue: "",   // optionally preload dialogue
      score: 0,       // optionally preload score
    };

    openScenePopup(scene, formId);
  }
});


function extractMovieCodeFromSceneId(sceneId) {
  return sceneId.split(".json")[0];
}

export function initSegmentModal() {
  if (document.getElementById("sceneModal")) return; // Prevent duplicates

  const modalHTML = `
    <div id="sceneModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background-color:rgba(0,0,0,0.8); z-index:1000;">
      <div id="modalContent">
        <span style="position:absolute; top:10px; right:15px; cursor:pointer; font-size:24px;" onclick="closeScenePopup()">&times;</span>
        <div id="modalInnerContent"></div>
      </div>
    </div>
  `;
  document.body.insertAdjacentHTML("beforeend", modalHTML);
  window.closeScenePopup = closeScenePopup; // expose for inline click

  // Close modal when clicking outside modal content
  document.getElementById("sceneModal").addEventListener("click", (e) => {
    if (e.target.id === "sceneModal") {
      closeScenePopup();
    }
  });
}


export function openScenePopup(scene,formId) {
  console.log("Scene data received:", scene);
  window.currentFormId = formId;
  window.currentSceneTitle = scene.title;
  window.currentSceneId = scene.scene_id;
  const {
    title,
    scene_id,
    moviecode,
    total_scenes,
    segment,
    type,
    desc,
    dialogue,
    location = [],
    emotion = {},
    season = "",
    episode = "",
    series = "",
    year = "",
    score = 0,
  } = scene;
  const [start, end] = segment;
  const sceneIdParsed = scene_id.replace(`${moviecode}.json - Scene `, "").trim();



  const modal = document.getElementById("sceneModal");
  const content = document.getElementById("modalInnerContent");

  const videoPath = `././movies/${moviecode}/movie.mp4`;
  const startSec = timeToSeconds(start);
  const endSec = timeToSeconds(end);

  if (isNaN(startSec) || isNaN(endSec)) {
    console.error("Invalid segment times:", start, end);
    alert("This scene has invalid time format and cannot be previewed.");
    return;
  }

  // Format locations
  const locationList = location.length
    ? location.map(loc => `<li>${loc.charAt(0).toUpperCase()}${loc.slice(1)}</li>`).join("")
    : "<li>None</li>";


  // Format emotion scores
  const emotionList = Object.values(emotion)
    .map(em => `<li>${em.charAt(0).toUpperCase()}${em.slice(1)}</li>`).join("");

  const seriesInfo =
    type === "series"
      ? `
    <p><span class="seriesTitle"> ${series}</span> — <span class="season">Season ${season}</span><span class="episode"> Episode ${episode}</span></p>
  `
      : "";

  content.innerHTML = `
    <div class="info-header">
      ${seriesInfo}
      <h3>${title} <span class="year">(${year})</span></h3>
      <p><span class="idScore"> <strong>Scene:</strong> ${sceneIdParsed}/${total_scenes}<strong>    Score:</strong> ${score.toFixed(3)}</span></p>
    </div>
    
    <div class="midSection">
      <div class="videoContainer">
        <video id="video" width="640" height="360" preload="metadata">
          <source src="${videoPath}" type="video/mp4" />
          <track src="././movies/${moviecode}/subtitles.vtt" kind="subtitles" srclang="en" label="English" default />
          Your browser does not support HTML5 video.
        </video>
      </div>

      <div class="info">
        <p><strong>Segment:</strong> ${start} – ${end}</p>
        <br>
        <p><strong>Description:</strong> ${desc}</p>
        <br>
        <p><strong>Dialogue:</strong> ${dialogue}</p>
        <br>
        <div class="locEmo">
          <div class="locations">
            <p><strong>Locations:</strong></p>
            <ul>${locationList}</ul>
          </div>
          <div class="emotions">
            <p><strong>Emotions:</strong></p>
            <ul>${emotionList}</ul>
          </div>
        </div>
      </div>
    </div>

    <div class="control-panel">
      <button class="paused" id="playPause"></button>
      <div class="slider-container" id="slider">
        <div class="range-bar" id="rangeBar"></div>
        <div class="handle" id="startHandle"></div>
        <div class="handle" id="endHandle"></div>
        <div class="playhead" id="playhead"></div>
      </div>
      <div class="time-display">
        <span id="startTime">${start}</span> |
        <span id="endTime">${end}</span>
      </div>
    </div>
    <button class="queueBtn" onclick='logSegment("${window.currentFormId}", ${JSON.stringify(window.currentSceneTitle)})'>Queue Scene</button>
  `;

  window.segment = { start: startSec, end: endSec };
  setupPopupPlayer();
  window.currentSceneId = scene_id;
  modal.style.display = "block";
}



function closeScenePopup() {
  const modal = document.getElementById("sceneModal");
  const content = document.getElementById("modalInnerContent");

  // Pause video and remove source to clear cache
  const video = content.querySelector("video");
  if (video) {
    video.pause();
    // Remove sources and reload video to release cache
    while (video.firstChild) {
      video.removeChild(video.firstChild);
    }
    video.load();
  }

  // Clear modal content and hide modal
  content.innerHTML = '';
  modal.style.display = "none";
}

function setupPopupPlayer() {
  const video = document.getElementById("video");
  const slider = document.getElementById("slider");
  const rangeBar = document.getElementById("rangeBar");
  const startHandle = document.getElementById("startHandle");
  const endHandle = document.getElementById("endHandle");
  const playhead = document.getElementById("playhead");
  const playPauseBtn = document.getElementById("playPause");
  const startTimeLabel = document.getElementById("startTime");
  const endTimeLabel = document.getElementById("endTime");

  let sliderWidth;
  let duration = 0;
  let windowStart, windowEnd, windowDuration;

  function timeToX(time) {
    return ((time - windowStart) / windowDuration) * sliderWidth;
  }

  function xToTime(x) {
    return windowStart + (x / sliderWidth) * windowDuration;
  }

  function updateHandles() {
    const startX = timeToX(window.segment.start);
    const endX = timeToX(window.segment.end);

    startHandle.style.left = `${startX}px`;
    endHandle.style.left = `${endX}px`;
    rangeBar.style.left = `${startX}px`;
    rangeBar.style.width = `${endX - startX}px`;

    startTimeLabel.textContent = `${secondsToTime(window.segment.start)}`;
    endTimeLabel.textContent = `${secondsToTime(window.segment.end)}`;
    updatePlayhead();
  }

  function updatePlayhead() {
    const currentX = timeToX(video.currentTime);
    playhead.style.left = `${currentX}px`;
  }

  function makeDraggable(handle, isStart) {
    let dragging = false;

    handle.addEventListener("mousedown", (e) => {
      dragging = true;
      e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const rect = slider.getBoundingClientRect();
      let x = e.clientX - rect.left;
      x = Math.max(0, Math.min(x, slider.offsetWidth));
      const time = xToTime(x);

      if (isStart) {
        window.segment.start = Math.min(time, window.segment.end - 1);
        video.currentTime = window.segment.start;
      } else {
        window.segment.end = Math.max(time, window.segment.start + 1);
      }
      updateHandles();
    });

    window.addEventListener("mouseup", () => dragging = false);
  }

  video.addEventListener("loadedmetadata", () => {
    duration = video.duration;
    sliderWidth = slider.offsetWidth;
    windowStart = Math.max(0, window.segment.start - windowMargin);
    windowEnd = Math.min(duration, window.segment.end + windowMargin);
    windowDuration = windowEnd - windowStart;

    updateHandles();
    video.currentTime = window.segment.start;
    updatePlayhead();
  });

  video.addEventListener("timeupdate", () => {
    if (video.currentTime > window.segment.end) {
      video.pause();
      playPauseBtn.classList.remove("playing");
      playPauseBtn.classList.add("paused");
    }
    updatePlayhead();
  });

  playPauseBtn.addEventListener("click", () => {
    if (video.paused) {
      // If we're at or past the segment end, restart at segment start
      if (video.currentTime >= window.segment.end) {
        video.currentTime = window.segment.start;
      }
      video.play();
      playPauseBtn.classList.add("playing");
      playPauseBtn.classList.remove("paused");
    } else {
      video.pause();
      playPauseBtn.classList.remove("playing");
      playPauseBtn.classList.add("paused");
    }
  });

  makeDraggable(startHandle, true);
  makeDraggable(endHandle, false);

  slider.addEventListener("click", (e) => {
    const rect = slider.getBoundingClientRect();
    let x = e.clientX - rect.left;
    x = Math.max(0, Math.min(x, slider.offsetWidth));
    const clickedTime = xToTime(x);
    video.currentTime = Math.max(window.segment.start, Math.min(clickedTime, window.segment.end));
    updatePlayhead();
  });
}

function timeToSeconds(t) {
  const [h, m, s] = t.split(":").map(Number);
  return h * 3600 + m * 60 + s;
}

function secondsToTime(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function replaceFormWithQueueMessage(formId, sceneTitle, thumbnails = []) {
  const wrapper = document.getElementById(`${formId}-wrapper`);
  if (!wrapper) return;

  let thumbsHTML = "";
  if (thumbnails.length) {
    thumbsHTML = `
      <div class="segment-thumbnails">
        ${thumbnails.map((src, idx) => `<img src="${src}" alt="Frame ${idx + 1}" width="280" />`).join("")}
      </div>
    `;
  }

  wrapper.innerHTML = `
    <div class="sceneDisplay">
      <p>${sceneTitle}</p>
      <button class="removeBtn" type="button" onclick="removeQueuedScene('${formId}', '${window.currentSceneId}')"></button>
      ${thumbsHTML}
      <button class="edit-scene-btn">Edit</button>
    </div>
  `;
}

window.removeQueuedScene = function(formId, sceneId) {
  // Remove from queuedScenes array
  const index = queuedScenes.indexOf(sceneId);
  if (index > -1) {
    queuedScenes.splice(index, 1);
  }

  // Optionally notify the server if needed
  fetch('/unqueue', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scene_id: sceneId })
  }).catch(err => {
    console.warn("Failed to notify server about unqueue:", err);
  });

  // Remove the wrapper from DOM
  const wrapper = document.getElementById(`${formId}-wrapper`);
  if (wrapper) wrapper.remove();

  // Optionally update stitch button state if needed
  const stitchBtn = document.getElementById(`${formId}-stitchBtn`);
  if (stitchBtn) stitchBtn.disabled = true;

  updateStatus(`Queued ${queuedScenes.length}/5 scenes.`, formId);
};


export function restoreSearchForm(formId) {
  const wrapper = document.getElementById(`${formId}-wrapper`);
  if (!wrapper) return;

  // Remove old wrapper (to reset event handlers etc)
  wrapper.remove();

  // Re-add a new search form with the same formId
  // Note: your addSearchForm() currently auto increments and creates new ids,
  // so let's make a helper to recreate a form with a given formId.

  createSearchFormWithId(formId);
}

window.logSegment = function(formId, sceneTitle) {
  console.log(`Selected segment: ${secondsToTime(window.segment.start)} - ${secondsToTime(window.segment.end)}`);

  const sceneId = window.currentSceneId;
  const wrapper = document.getElementById(`${formId}-wrapper`);
  wrapper.classList.add('queued');
  wrapper.dataset.sceneId = sceneId;
  wrapper.dataset.start_time = window.segment.start;
  wrapper.dataset.end_time = window.segment.end;

  const video = document.getElementById("video");
  const start = window.segment.start;
  const end = window.segment.end;
  const mid = start + (end - start) / 2;

  const captureFrameAt = (time) => {
    return new Promise((resolve) => {
      video.currentTime = time;
      const handler = () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        video.removeEventListener("seeked", handler);
        resolve(canvas.toDataURL("image/jpeg"));
      };
      video.addEventListener("seeked", handler);
    });
  };

  const captureAndDisplay = async () => {
    try {
      const segmentLength = end - start;
      const timestamps = [
        start,
        start + 0.25 * segmentLength,
        start + 0.75 * segmentLength,
        end
      ];

      const frames = [];

      for (const time of timestamps) {
        const frame = await captureFrameAt(time);
        frames.push(frame);
      }

      closeScenePopup();
      replaceFormWithQueueMessage(formId, sceneTitle, frames);
    } catch (err) {
      console.error("Frame capture failed:", err);
      closeScenePopup();
      replaceFormWithQueueMessage(formId, sceneTitle);
    }
  };



  captureAndDisplay();
};


window.onSegmentQueued = function(sceneId, formId = null) {
  if (!queuedScenes.includes(sceneId)) {
    queuedScenes.push(sceneId);
    updateStatus(`Queued ${queuedScenes.length}/5 scenes.`, formId);
    if (formId) {
      const stitchBtn = document.getElementById(`${formId}-stitchBtn`);
      if (stitchBtn) {
        stitchBtn.disabled = false;
      }
    } else {
      // fallback if no formId
      const stitchBtn = document.getElementById('stitchBtn');
      if (stitchBtn) stitchBtn.disabled = false;
    }
  }
};

