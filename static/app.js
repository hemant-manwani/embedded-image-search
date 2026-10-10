// static/app.js - Fullscreen Masonry Semantic Search Controller

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const searchForm = document.getElementById("search-form");
  const searchInput = document.getElementById("search-input");
  const searchSubmitBtn = document.getElementById("search-submit-btn");
  const masonryGrid = document.getElementById("masonry-grid");
  const emptyState = document.getElementById("empty-state");
  const searchLoader = document.getElementById("search-loader");
  const loaderText = document.getElementById("loader-text");
  
  // Floating Active Query Pill
  const searchActivePill = document.getElementById("search-active-pill");
  const searchActiveQuery = document.getElementById("search-active-query");
  const searchActiveStats = document.getElementById("search-active-stats");
  const clearSearchBtn = document.getElementById("clear-search-btn");
  
  // Header Elements
  const navDbCount = document.getElementById("nav-db-count");
  
  // Lightbox Dialog Elements
  const lightboxDialog = document.getElementById("lightbox-dialog");
  const lightboxCloseBtn = document.getElementById("lightbox-close-btn");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxSampleTag = document.getElementById("lightbox-sample-tag");
  const lightboxMatchBadge = document.getElementById("lightbox-match-badge");
  const lightboxPrimaryCaption = document.getElementById("lightbox-primary-caption");
  const lightboxCaptionsList = document.getElementById("lightbox-captions-list");
  const lightboxRawLink = document.getElementById("lightbox-raw-link");

  let isSearching = false;

  // 1. Initial Load: Gallery Preview & Database Stats
  loadInitialGallery();
  fetchNavStats();
  setInterval(fetchNavStats, 6000);

  // 2. Search Form Submit
  searchForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = searchInput.value.trim();
    if (query && !isSearching) {
      executeSearch(query);
    }
  });

  // 3. Quick Suggestion Chips
  document.querySelectorAll(".dock-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.dataset.query;
      searchInput.value = q;
      searchInput.focus();
      executeSearch(q);
    });
  });

  // 4. Clear Search Pill (Reset to Gallery)
  clearSearchBtn.addEventListener("click", () => {
    searchInput.value = "";
    searchActivePill.classList.add("hidden");
    loadInitialGallery();
  });

  // 5. Global Keyboard Shortcuts: '/' to focus search, 'Esc' to close lightbox
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== searchInput && !lightboxDialog.open) {
      e.preventDefault();
      searchInput.focus();
      searchInput.select();
    }
    if (e.key === "Escape" && lightboxDialog.open) {
      closeLightbox();
    }
  });

  // 6. Lightbox Event Handlers
  lightboxCloseBtn.addEventListener("click", closeLightbox);
  lightboxDialog.addEventListener("click", (e) => {
    // Close if clicking outside container (on the backdrop)
    const rect = lightboxDialog.getBoundingClientRect();
    const isInDialog = (rect.top <= e.clientY && e.clientY <= rect.top + rect.height
      && rect.left <= e.clientX && e.clientX <= rect.left + rect.width);
    if (!isInDialog) {
      closeLightbox();
    }
  });

  function openLightbox(item) {
    lightboxImg.src = item.image_url;
    lightboxSampleTag.textContent = `Sample #${item.sample_idx}`;
    
    if (item.match_percentage !== undefined) {
      lightboxMatchBadge.textContent = `${item.match_percentage}% Match`;
      lightboxMatchBadge.style.display = "inline-block";
    } else {
      lightboxMatchBadge.style.display = "none";
    }

    const primaryCaption = item.caption || (item.captions && item.captions[0]) || "No caption available.";
    lightboxPrimaryCaption.textContent = primaryCaption;

    // Populate all Flickr30k reference captions
    const captions = item.captions || [primaryCaption];
    lightboxCaptionsList.innerHTML = captions.map((c) => `<li>${escapeHtml(c)}</li>`).join("");

    lightboxRawLink.href = item.image_url;
    lightboxDialog.showModal();
  }

  function closeLightbox() {
    lightboxDialog.close();
  }

  // 7. Load Gallery Preview
  async function loadInitialGallery() {
    try {
      emptyState.classList.add("hidden");
      const res = await fetch("/api/gallery?count=30");
      if (!res.ok) return;
      const data = await res.json();
      renderCards(data.gallery || [], false);
    } catch (err) {
      console.error("Failed to load initial gallery:", err);
    }
  }

  // 8. Execute Semantic Search
  async function executeSearch(query) {
    isSearching = true;
    searchLoader.classList.remove("hidden");
    loaderText.textContent = `Embedding "${query}" & scanning LanceDB...`;
    searchSubmitBtn.style.opacity = "0.5";
    emptyState.classList.add("hidden");

    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&top_k=24`);
      if (!res.ok) throw new Error("Search request failed");
      const data = await res.json();

      const results = data.results || [];

      // Update Floating Active Query Pill
      searchActiveQuery.textContent = `"${data.query}"`;
      searchActiveStats.textContent = `${results.length} matches (${data.timing_ms.total_latency}ms • LanceDB: ${data.timing_ms.vector_search}ms)`;
      searchActivePill.classList.remove("hidden");

      if (results.length === 0) {
        masonryGrid.innerHTML = "";
        emptyState.classList.remove("hidden");
      } else {
        renderCards(results, true);
        window.scrollTo({ top: 0, behavior: "smooth" });
      }

    } catch (err) {
      console.error("Search error:", err);
      alert(`Search failed: ${err.message}`);
    } finally {
      isSearching = false;
      searchLoader.classList.add("hidden");
      searchSubmitBtn.style.opacity = "1";
    }
  }

  // 9. Render Cards with Masonry Stagger Animation
  function renderCards(items, isSearchResult) {
    masonryGrid.innerHTML = "";

    items.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "masonry-card";
      // Pass staggered animation delay to CSS
      card.style.setProperty("--delay", Math.min(index, 20));

      const matchBadgeHtml = isSearchResult && item.match_percentage !== undefined
        ? `<span class="match-badge">${item.match_percentage}%</span>`
        : "";

      const rankBadgeHtml = isSearchResult && item.rank !== undefined
        ? `<span class="rank-badge">#${item.rank}</span>`
        : `<span class="rank-badge">#${item.sample_idx}</span>`;

      const captionText = item.caption || (item.captions && item.captions[0]) || "";

      card.innerHTML = `
        <div class="card-img-holder">
          <img src="${item.image_url}" alt="Sample #${item.sample_idx}" loading="lazy">
          <div class="card-badges-row">
            ${rankBadgeHtml}
            ${matchBadgeHtml}
          </div>
          <div class="card-scrim">
            <p class="card-caption-text">"${escapeHtml(captionText)}"</p>
            <span class="card-sample-id">Flickr30k #${item.sample_idx}</span>
          </div>
        </div>
      `;

      // Open lightbox on card click
      card.addEventListener("click", () => {
        openLightbox(item);
      });

      masonryGrid.appendChild(card);
    });
  }

  // 10. Fetch Navigation Stats
  async function fetchNavStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) return;
      const data = await res.json();
      if (navDbCount) {
        navDbCount.textContent = `LanceDB (${data.indexed_images.toLocaleString()} images)`;
      }
    } catch (err) {
      console.warn("Could not fetch stats:", err);
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
});
