// static/manage.js - LanceDB Vector Index Manager Controller

document.addEventListener("DOMContentLoaded", () => {
  // KPI Elements
  const kpiIndexedCount = document.getElementById("kpi-indexed-count");
  const kpiCoveragePct = document.getElementById("kpi-coverage-pct");
  const kpiDiskSize = document.getElementById("kpi-disk-size");
  const globalStatusDot = document.getElementById("global-status-dot");
  const globalStatusText = document.getElementById("global-status-text");

  // Controller Elements
  const jobStatusBadge = document.getElementById("job-status-badge");
  const targetInput = document.getElementById("target-input");
  const startBtn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const progressMessage = document.getElementById("progress-message");
  const progressPercent = document.getElementById("progress-percent");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const metaSpeed = document.getElementById("meta-speed");
  const metaEta = document.getElementById("meta-eta");
  const metaRemaining = document.getElementById("meta-remaining");

  // Live Inspect Elements
  const liveInspectCard = document.getElementById("live-inspect-card");
  const liveThumb = document.getElementById("live-thumb");
  const liveSampleId = document.getElementById("live-sample-id");
  const liveCaption = document.getElementById("live-caption");

  // Table Inspector Elements
  const recordsTableBody = document.getElementById("records-table-body");
  const refreshRecordsBtn = document.getElementById("refresh-records-btn");

  let pollInterval = null;
  let isIndexing = false;
  let currentIndexed = 0;

  // 1. Initial Load
  fetchStats();
  fetchStatus();
  fetchRecentRecords();
  startPolling(1500);

  // 2. Preset Buttons (+50, +100, etc.)
  document.querySelectorAll(".btn-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      const add = parseInt(btn.dataset.add, 10);
      const base = Math.max(currentIndexed, parseInt(targetInput.value, 10) || currentIndexed);
      targetInput.value = base + add;
    });
  });

  // 3. Start Indexing
  startBtn.addEventListener("click", async () => {
    const target = parseInt(targetInput.value, 10);
    if (!target || target <= 0) {
      alert("Please enter a valid target image count.");
      return;
    }

    if (target <= currentIndexed) {
      const proceed = confirm(`LanceDB already has ${currentIndexed} images. Target ${target} is <= current count. Increase target?`);
      if (proceed) {
        targetInput.value = currentIndexed + 50;
        return;
      }
    }

    try {
      startBtn.disabled = true;
      jobStatusBadge.className = "badge badge-active";
      jobStatusBadge.textContent = "Starting...";
      
      const res = await fetch(`/api/index/start?target=${encodeURIComponent(targetInput.value)}`, {
        method: "POST"
      });
      const data = await res.json();
      if (data.status === "started") {
        isIndexing = true;
        stopBtn.disabled = false;
        fetchStatus();
      }
    } catch (err) {
      console.error("Failed to start indexing:", err);
      startBtn.disabled = false;
    }
  });

  // 4. Stop Indexing
  stopBtn.addEventListener("click", async () => {
    try {
      stopBtn.disabled = true;
      jobStatusBadge.textContent = "Pausing...";
      await fetch("/api/index/stop", { method: "POST" });
    } catch (err) {
      console.error("Failed to stop indexing:", err);
    }
  });

  // 5. Refresh Records Button
  refreshRecordsBtn.addEventListener("click", () => {
    fetchRecentRecords();
    fetchStats();
  });

  // Polling loop
  function startPolling(ms) {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(fetchStatus, ms);
  }

  // Fetch KPI Stats
  async function fetchStats() {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();
      currentIndexed = data.indexed_images;

      kpiIndexedCount.textContent = data.indexed_images.toLocaleString();
      kpiDiskSize.textContent = `${data.db_size_mb} MB`;

      const coverage = ((data.indexed_images / data.total_dataset_images) * 100).toFixed(2);
      kpiCoveragePct.textContent = `${coverage}% of dataset indexed`;

      // Set default target input if blank
      if (!targetInput.value || parseInt(targetInput.value, 10) <= currentIndexed) {
        targetInput.value = currentIndexed + 100;
      }
    } catch (err) {
      console.error("Failed to fetch stats:", err);
    }
  }

  // Fetch Index Status
  async function fetchStatus() {
    try {
      const res = await fetch("/api/index/status");
      const status = await res.json();

      isIndexing = status.is_indexing;
      currentIndexed = status.current_count;
      kpiIndexedCount.textContent = status.current_count.toLocaleString();
      kpiDiskSize.textContent = `${status.index_file_size_mb} MB`;

      const coverage = ((status.current_count / status.total_available) * 100).toFixed(2);
      kpiCoveragePct.textContent = `${coverage}% of dataset indexed`;

      // Update UI Controls based on indexing state
      if (status.is_indexing) {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        jobStatusBadge.className = "badge badge-active";
        jobStatusBadge.textContent = "Indexing Active";

        globalStatusDot.className = "status-dot status-indexing";
        globalStatusText.textContent = `Indexing (${status.current_count}/${status.target_count})`;

        // Live stream inspect
        if (status.current_sample_idx !== null) {
          liveInspectCard.classList.remove("hidden");
          const imgUrl = `/api/image/${status.current_sample_idx}`;
          if (liveThumb.src !== window.location.origin + imgUrl) {
            liveThumb.src = imgUrl;
          }
          liveSampleId.textContent = `Sample #${status.current_sample_idx}`;
          liveCaption.textContent = status.current_caption || "Encoding multimodal features...";
        }

        // Progress bar
        const pct = Math.min(100, Math.max(0, status.progress_percent));
        progressBarFill.style.width = `${pct}%`;
        progressPercent.textContent = `${pct}%`;
        progressMessage.textContent = status.message || `Indexing images into LanceDB...`;

        // Metrics
        const spd = status.speed_sec_per_img > 0 ? status.speed_sec_per_img.toFixed(1) : "--";
        metaSpeed.textContent = `Speed: ${spd}s / img`;

        const remaining = Math.max(0, status.target_count - status.current_count);
        metaRemaining.textContent = `Remaining: ${remaining} images`;

        if (status.speed_sec_per_img > 0 && remaining > 0) {
          const etaSec = Math.round(remaining * status.speed_sec_per_img);
          const etaMin = Math.floor(etaSec / 60);
          const etaSecRem = etaSec % 60;
          metaEta.textContent = etaMin > 0 ? `ETA: ~${etaMin}m ${etaSecRem}s` : `ETA: ~${etaSecRem}s`;
        } else {
          metaEta.textContent = `ETA: Calculating...`;
        }
      } else {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        jobStatusBadge.className = "badge badge-idle";
        jobStatusBadge.textContent = "Worker Idle";

        globalStatusDot.className = "status-dot";
        globalStatusText.textContent = "Ready";

        liveInspectCard.classList.add("hidden");

        if (status.message && status.message !== "Idle") {
          progressMessage.textContent = status.message;
        } else {
          progressMessage.textContent = "Worker is idle. Choose a target count to begin.";
        }

        if (status.target_count > 0 && status.current_count >= status.target_count) {
          progressBarFill.style.width = "100%";
          progressPercent.textContent = "100%";
        }

        metaSpeed.textContent = "Speed: --";
        metaEta.textContent = "ETA: --";
        metaRemaining.textContent = "Remaining: 0";
      }
    } catch (err) {
      console.error("Failed to fetch index status:", err);
    }
  }

  // Fetch Recent LanceDB Records
  async function fetchRecentRecords() {
    try {
      const res = await fetch("/api/index/records?limit=10");
      const data = await res.json();
      const records = data.records || [];

      if (records.length === 0) {
        recordsTableBody.innerHTML = `
          <tr>
            <td colspan="5" class="table-empty">No records indexed in LanceDB yet.</td>
          </tr>
        `;
        return;
      }

      recordsTableBody.innerHTML = records.map((rec) => {
        const previewStr = rec.vector_preview.length > 0 
          ? `[${rec.vector_preview.join(", ")}, …]` 
          : "[768 float32 dimensions]";

        return `
          <tr class="record-row">
            <td>
              <img class="table-thumb" src="${rec.image_url}" alt="Sample #${rec.sample_idx}" loading="lazy">
            </td>
            <td>
              <span class="sample-badge">#${rec.sample_idx}</span>
            </td>
            <td>
              <div class="caption-text" title="${escapeHtml(rec.caption)}">${escapeHtml(rec.caption)}</div>
            </td>
            <td>
              <div class="vector-code" title="768-D unit vector preview">
                <span class="dim-badge">768-D</span>
                <code>${previewStr}</code>
              </div>
            </td>
            <td style="text-align: right;">
              <a href="${rec.image_url}" target="_blank" class="btn-table-view">View</a>
            </td>
          </tr>
        `;
      }).join("");

    } catch (err) {
      console.error("Failed to fetch records:", err);
      recordsTableBody.innerHTML = `
        <tr>
          <td colspan="5" class="table-error">Failed to load LanceDB records.</td>
        </tr>
      `;
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
});
