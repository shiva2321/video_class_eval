// 8x8 Class Quality Assessor UI Client
(function () {
  const CHUNK_SIZE = 8 * 1024 * 1024; // 8 MiB
  const STAGES = [
    "queued",
    "ingest",
    "asr",
    "diarize",
    "audio",
    "video",
    "understand",
    "score",
    "report",
    "done"
  ];

  // DOM Elements
  const accessCodeInput = document.getElementById("accessCode");
  const toggleCodeBtn = document.getElementById("toggleCodeBtn");
  const uploadSection = document.getElementById("uploadSection");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const fileInfo = document.getElementById("fileInfo");
  const fileName = document.getElementById("fileName");
  const fileSize = document.getElementById("fileSize");
  const startBtn = document.getElementById("startBtn");
  const progressSection = document.getElementById("progressSection");
  const progressStatusText = document.getElementById("progressStatusText");
  const progressPercent = document.getElementById("progressPercent");
  const progressBar = document.getElementById("progressBar");
  const errorCard = document.getElementById("errorCard");
  const errorTitle = document.getElementById("errorTitle");
  const errorMessage = document.getElementById("errorMessage");
  const dismissErrorBtn = document.getElementById("dismissErrorBtn");
  const reportSection = document.getElementById("reportSection");
  const reportFrame = document.getElementById("reportFrame");
  const newAssessmentBtn = document.getElementById("newAssessmentBtn");

  let selectedFile = null;
  let currentEventSource = null;

  // A shared link may carry the access code after the # (the address followed by #code=and the code). The part after # never goes to the
  // server; it is moved into the field and removed from the address bar so it is not copied by accident.
  const linkCode = (new URLSearchParams(location.hash.slice(1)).get("code") || "").trim();
  if (/^[!-~]{1,128}$/.test(linkCode)) {  // printable characters only, bounded length; anything else is ignored
    localStorage.setItem("classeval_access_code", linkCode);
    history.replaceState(null, "", location.pathname + location.search);
  }

  // Restore access code from cookie or localStorage
  const savedCode = localStorage.getItem("classeval_access_code") || getCookie("access");
  if (savedCode) {
    accessCodeInput.value = savedCode;
    setCookie("access", savedCode, 7);
  }

  accessCodeInput.addEventListener("input", function () {
    const val = accessCodeInput.value.trim();
    localStorage.setItem("classeval_access_code", val);
    setCookie("access", val, 7);
  });

  toggleCodeBtn.addEventListener("click", function () {
    if (accessCodeInput.type === "password") {
      accessCodeInput.type = "text";
      toggleCodeBtn.textContent = "Hide";
    } else {
      accessCodeInput.type = "password";
      toggleCodeBtn.textContent = "Show";
    }
  });

  // Drag and Drop & File Selection
  browseBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", function () {
    fileInput.click();
  });

  dropzone.addEventListener("dragover", function (e) {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", function () {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", function () {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    fileInfo.classList.remove("hidden");
    hideError();
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  dismissErrorBtn.addEventListener("click", hideError);

  function showError(title, msg) {
    errorTitle.textContent = title;
    errorMessage.textContent = msg;
    errorCard.classList.remove("hidden");
  }

  function hideError() {
    errorCard.classList.add("hidden");
  }

  newAssessmentBtn.addEventListener("click", function () {
    reportSection.classList.add("hidden");
    progressSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");
    fileInfo.classList.add("hidden");
    selectedFile = null;
    fileInput.value = "";
    resetStages();
  });

  function resetStages() {
    document.querySelectorAll(".stage-item").forEach(function (el) {
      el.classList.remove("active", "completed");
    });
    progressBar.style.width = "0%";
    progressPercent.textContent = "0%";
  }

  function updateStageUI(activeStage, progress) {
    const activeIdx = STAGES.indexOf(activeStage);
    document.querySelectorAll(".stage-item").forEach(function (el) {
      const stageName = el.getAttribute("data-stage");
      const idx = STAGES.indexOf(stageName);
      if (idx < activeIdx) {
        el.classList.remove("active");
        el.classList.add("completed");
      } else if (idx === activeIdx) {
        el.classList.add("active");
        el.classList.remove("completed");
      } else {
        el.classList.remove("active", "completed");
      }
    });

    const pct = Math.round(progress * 100);
    progressBar.style.width = pct + "%";
    progressPercent.textContent = pct + "%";
    progressStatusText.textContent = "Processing: " + activeStage;
  }

  function getAuthHeader() {
    const code = accessCodeInput.value.trim();
    return { "X-Access-Code": code };
  }

  function friendlyErrorMessage(status, defaultMsg) {
    if (status === 401) {
      return "Access denied. Please enter a valid access code.";
    }
    if (status === 413) {
      return "The recording file exceeds the upload limit (2.5 GB).";
    }
    if (status === 429) {
      return "Rate limit reached. Please wait 60 seconds before making more requests.";
    }
    if (status >= 500) {
      return "Server error occurred. Please check the server logs.";
    }
    return defaultMsg || "An unexpected error occurred.";
  }

  // Upload and Job Execution
  startBtn.addEventListener("click", async function () {
    if (!selectedFile) {
      showError("File Required", "Please select a recording file to assess.");
      return;
    }
    const code = accessCodeInput.value.trim();
    if (!code) {
      showError("Access Code Required", "Please enter the access code to proceed.");
      return;
    }

    uploadSection.classList.add("hidden");
    progressSection.classList.remove("hidden");
    hideError();
    resetStages();

    try {
      progressStatusText.textContent = "Initializing upload...";
      const initResp = await fetch("/api/uploads", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeader()
        },
        body: JSON.stringify({
          size: selectedFile.size,
          filename: selectedFile.name
        })
      });

      if (!initResp.ok) {
        throw new Error(friendlyErrorMessage(initResp.status, "Failed to initialize upload."));
      }

      const initData = await initResp.json();
      const uploadId = initData.upload_id;

      // Upload Chunks
      await uploadChunks(uploadId, selectedFile);

      // Complete Upload
      progressStatusText.textContent = "Finalizing upload...";
      const completeResp = await fetch("/api/uploads/" + uploadId + "/complete", {
        method: "POST",
        headers: getAuthHeader()
      });

      if (!completeResp.ok) {
        if (completeResp.status === 409) {
          const compData = await completeResp.json();
          throw new Error("Upload missing chunks: " + JSON.stringify(compData.missing));
        }
        throw new Error(friendlyErrorMessage(completeResp.status, "Failed to complete upload."));
      }

      // Create Job
      progressStatusText.textContent = "Queueing job...";
      const jobResp = await fetch("/api/jobs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeader()
        },
        body: JSON.stringify({
          type: "upload",
          upload_id: uploadId
        })
      });

      if (!jobResp.ok) {
        throw new Error(friendlyErrorMessage(jobResp.status, "Failed to create assessment job."));
      }

      const jobData = await jobResp.json();
      const jobId = jobData.job_id;

      // Subscribe to Events
      subscribeToJob(jobId);
    } catch (err) {
      showError("Upload / Execution Failed", err.message);
      progressSection.classList.add("hidden");
      uploadSection.classList.remove("hidden");
    }
  });

  async function uploadChunks(uploadId, file) {
    const totalBytes = file.size;
    const totalChunks = Math.max(1, Math.ceil(totalBytes / CHUNK_SIZE));

    for (let i = 0; i < totalChunks; i++) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, totalBytes);
      const chunk = file.slice(start, end);

      let success = false;
      let retries = 3;
      let delay = 500;

      while (!success && retries > 0) {
        try {
          const chunkPct = Math.round(((i) / totalChunks) * 100);
          progressBar.style.width = chunkPct + "%";
          progressPercent.textContent = chunkPct + "%";
          progressStatusText.textContent = "Uploading chunk " + (i + 1) + " of " + totalChunks;

          const resp = await fetch("/api/uploads/" + uploadId + "/chunks/" + i, {
            method: "PUT",
            headers: {
              "Content-Type": "application/octet-stream",
              ...getAuthHeader()
            },
            body: chunk
          });

          if (resp.ok) {
            success = true;
          } else {
            retries--;
            if (retries === 0) {
              throw new Error(friendlyErrorMessage(resp.status, "Failed to upload chunk " + i));
            }
            await new Promise(function (res) { setTimeout(res, delay); });
            delay *= 2;
          }
        } catch (e) {
          retries--;
          if (retries === 0) throw e;
          await new Promise(function (res) { setTimeout(res, delay); });
          delay *= 2;
        }
      }
    }
    progressBar.style.width = "100%";
    progressPercent.textContent = "100%";
  }

  function subscribeToJob(jobId) {
    if (currentEventSource) {
      currentEventSource.close();
    }

    const eventUrl = "/api/jobs/" + jobId + "/events";
    const es = new EventSource(eventUrl);
    currentEventSource = es;

    es.addEventListener("stage", function (event) {
      try {
        const data = JSON.parse(event.data);
        updateStageUI(data.stage, data.progress);

        if (data.status === "done") {
          es.close();
          showReport(jobId);
        } else if (data.status === "failed") {
          es.close();
          const errDetail = data.error ? (data.error.message || JSON.stringify(data.error)) : "Pipeline failed.";
          showError("Assessment Failed", errDetail);
        }
      } catch (e) {
        console.error("Failed to parse stage event:", e);
      }
    });

    es.onerror = function () {
      // If disconnected or closed, poll status as fallback
      pollJobStatus(jobId);
    };
  }

  function pollJobStatus(jobId) {
    let failures = 0;
    const pollInterval = setInterval(async function () {
      try {
        const resp = await fetch("/api/jobs/" + jobId, { headers: getAuthHeader() });
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        failures = 0;
        const job = await resp.json();
        updateStageUI(job.stage, job.progress);
        if (job.status === "done") {
          clearInterval(pollInterval);
          if (currentEventSource) currentEventSource.close();
          showReport(jobId);
        } else if (job.status === "failed") {
          clearInterval(pollInterval);
          if (currentEventSource) currentEventSource.close();
          showError("Assessment Failed", job.error ? job.error.message : "Pipeline failed.");
        }
      } catch (e) {
        failures += 1;
        console.warn("Status poll failed (" + failures + " in a row):", e);
        if (failures >= 5) {
          clearInterval(pollInterval);
          showError("Connection lost", "Could not reach the server to check progress. Reload the page and try again.");
        }
      }
    }, 2000);
  }

  function showReport(jobId) {
    progressSection.classList.add("hidden");
    reportSection.classList.remove("hidden");
    reportFrame.src = "/api/jobs/" + jobId + "/report.html";
    document.getElementById("openReportLink").href = "/api/jobs/" + jobId + "/report.html";
  }

  function setCookie(name, value, days) {
    let expires = "";
    if (days) {
      const date = new Date();
      date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
      expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + encodeURIComponent(value || "") + expires + "; path=/; SameSite=Strict" + (location.protocol === "https:" ? "; Secure" : "");
  }

  function getCookie(name) {
    const nameEQ = name + "=";
    const ca = document.cookie.split(";");
    for (let i = 0; i < ca.length; i++) {
      let c = ca[i];
      while (c.charAt(0) === " ") c = c.substring(1, c.length);
      if (c.indexOf(nameEQ) === 0) {
        const raw = c.substring(nameEQ.length, c.length);
        try {
          return decodeURIComponent(raw);
        } catch (e) {
          return raw;
        }
      }
    }
    return null;
  }
})();
