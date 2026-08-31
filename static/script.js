// script.js

document.addEventListener("DOMContentLoaded", () => {
    // Icons initialization function
    const refreshIcons = () => {
        if (window.lucide) {
            window.lucide.createIcons();
        }
    };

    // ==========================================
    // Theme Management
    // ==========================================
    const themeToggleBtn = document.getElementById("theme-toggle");
    const htmlEl = document.documentElement;

    // Load saved theme
    const savedTheme = localStorage.getItem("theme") || "dark";
    htmlEl.setAttribute("data-theme", savedTheme);

    themeToggleBtn.addEventListener("click", () => {
        const currentTheme = htmlEl.getAttribute("data-theme");
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        htmlEl.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
        refreshIcons();
    });

    // ==========================================
    // Navigation Tabs
    // ==========================================
    const tabBtns = document.querySelectorAll(".nav-tabs .tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");

            // Deactivate all
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            // Activate target
            btn.classList.add("active");
            document.getElementById(`tab-${targetTab}`).classList.add("active");
            refreshIcons();
        });
    });

    // ==========================================
    // Collapsible Settings
    // ==========================================
    const advToggle = document.getElementById("advanced-toggle");
    const advPanel = document.getElementById("advanced-settings-panel");

    advToggle.addEventListener("click", () => {
        advPanel.classList.toggle("hidden");
        const isHidden = advPanel.classList.contains("hidden");
        advToggle.classList.toggle("active", !isHidden);
    });

    // ==========================================
    // Single Comment Classifier
    // ==========================================
    const commentInput = document.getElementById("comment-input");
    const langSelect = document.getElementById("lang-select");
    const featureSelect = document.getElementById("feature-select");
    const modelSelect = document.getElementById("model-select");

    const btnClear = document.getElementById("btn-clear");
    const btnAnalyze = document.getElementById("btn-analyze");

    const resultsPanel = document.getElementById("results-panel");
    const badgeStatus = document.getElementById("badge-status");
    const badgeLang = document.getElementById("badge-detected-lang");
    const gaugeFill = document.getElementById("gauge-fill");
    const toxicityPercent = document.getElementById("toxicity-percent");
    const categoriesGrid = document.getElementById("categories-grid");
    const highlightOutput = document.getElementById("highlight-output");
    const suggestionsBox = document.getElementById("suggestions-box");
    const suggestionsList = document.getElementById("suggestions-list");

    const categoriesList = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"];

    btnClear.addEventListener("click", () => {
        commentInput.value = "";
        resultsPanel.classList.add("hidden");
    });

    btnAnalyze.addEventListener("click", async () => {
        const text = commentInput.value.trim();
        if (!text) {
            alert("Please enter a comment to analyze.");
            return;
        }

        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = `<i data-lucide="loader-2" class="animate-spin"></i> Analyzing...`;
        refreshIcons();

        try {
            const response = await fetch("/api/classify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    comment: text,
                    language: langSelect.value,
                    feature_mode: featureSelect.value,
                    model_type: modelSelect.value
                })
            });

            if (!response.ok) {
                throw new Error("Prediction API request failed.");
            }

            const data = await response.json();
            renderPredictionResults(text, data);

        } catch (error) {
            console.error(error);
            alert("Error running toxic comment analysis. Please check server logs.");
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = `<i data-lucide="shield-alert"></i> Analyze Comment`;
            refreshIcons();
        }
    });

    function renderPredictionResults(rawText, data) {
        resultsPanel.classList.remove("hidden");

        // Set Clean/Toxic status
        if (data.is_toxic) {
            badgeStatus.className = "badge badge-toxic";
            badgeStatus.textContent = "⚠️ Flagged Toxic";
        } else {
            badgeStatus.className = "badge badge-clean";
            badgeStatus.textContent = "✅ Clean / Safe";
        }

        // Language
        badgeLang.textContent = `🌐 ${data.detected_language}`;

        // Set Gauge Severity & color
        const pct = Math.round(data.overall_score * 100);
        gaugeFill.style.width = `${pct}%`;
        toxicityPercent.textContent = `${pct}%`;

        if (pct < 45) {
            gaugeFill.style.backgroundColor = "var(--success)";
        } else if (pct < 75) {
            gaugeFill.style.backgroundColor = "var(--warning)";
        } else {
            gaugeFill.style.backgroundColor = "var(--danger)";
        }

        // Render Category Pills
        categoriesGrid.innerHTML = "";
        categoriesList.forEach(cat => {
            const prob = data.probabilities[cat] || 0.0;
            const isFlagged = data.binary_predictions[cat] === 1;
            
            const pill = document.createElement("div");
            pill.className = `category-pill ${isFlagged ? "active" : ""}`;
            pill.innerHTML = `
                <div style="font-size: 0.72rem; opacity: 0.8; text-transform: uppercase;">${cat.replace('_', ' ')}</div>
                <div style="font-size: 1.05rem; font-weight: 700; margin-top: 2px;">${Math.round(prob * 100)}%</div>
            `;
            categoriesGrid.appendChild(pill);
        });

        // Set Highlighted Spans
        highlightOutput.innerHTML = data.highlighted_html || rawText;

        // Polite Suggestions
        suggestionsList.innerHTML = "";
        if (data.suggestions && data.suggestions.length > 0) {
            suggestionsBox.classList.remove("hidden");
            data.suggestions.forEach(s => {
                const item = document.createElement("div");
                item.style.marginBottom = "8px";
                item.innerHTML = `
                    <div style="font-weight: 600; font-size: 0.8rem; color: var(--success); margin-bottom: 2px;">Instead of: "${s.toxic_phrase}"</div>
                    <div style="font-style: italic; font-size: 0.95rem;">👉 "${s.polite_suggestion}"</div>
                `;
                suggestionsList.appendChild(item);
            });
        } else {
            suggestionsBox.classList.add("hidden");
        }
    }

    // ==========================================
    // Batch Moderation Upload Handlers
    // ==========================================
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const batchSettings = document.getElementById("batch-settings");
    const columnSelect = document.getElementById("column-select");
    const btnBatchProcess = document.getElementById("btn-batch-process");
    const batchResults = document.getElementById("batch-results");

    const batchTotal = document.getElementById("batch-total");
    const batchFlagged = document.getElementById("batch-flagged");
    const batchClean = document.getElementById("batch-clean");
    const batchTableBody = document.querySelector("#batch-preview-table tbody");
    const btnDownloadReport = document.getElementById("btn-download-report");

    let currentFile = null;
    let fileHeaders = [];
    let processedCSVContent = "";

    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // Drag-and-drop actions
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        handleFileSelect(e.dataTransfer.files[0]);
    });

    function handleFileSelect(file) {
        if (!file || !file.name.endsWith(".csv")) {
            alert("Please upload a valid CSV file.");
            return;
        }

        currentFile = file;
        dropZone.querySelector("p").innerHTML = `Selected: <strong>${file.name}</strong> (${Math.round(file.size / 1024)} KB)`;
        
        // Parse CSV headers
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            const firstLine = text.split("\n")[0];
            fileHeaders = firstLine.split(",").map(h => h.trim().replace(/^["']|["']$/g, ''));
            
            columnSelect.innerHTML = "";
            fileHeaders.forEach(header => {
                const opt = document.createElement("option");
                opt.value = header;
                opt.textContent = header;
                columnSelect.appendChild(opt);
            });

            batchSettings.classList.remove("hidden");
            batchResults.classList.add("hidden");
        };
        reader.readAsText(file);
    }

    btnBatchProcess.addEventListener("click", async () => {
        if (!currentFile) return;

        btnBatchProcess.disabled = true;
        btnBatchProcess.innerHTML = `<i data-lucide="loader-2" class="animate-spin"></i> Processing...`;
        refreshIcons();

        const formData = new FormData();
        formData.append("file", currentFile);
        formData.append("comment_column", columnSelect.value);
        formData.append("feature_mode", featureSelect.value);
        formData.append("model_type", modelSelect.value);

        try {
            const response = await fetch("/api/moderate_file", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error("Batch processing failed.");
            }

            const data = await response.json();
            processedCSVContent = data.csv_content;
            renderBatchResults(data.rows);

        } catch (error) {
            console.error(error);
            alert("Error during batch CSV moderation. Please verify the CSV columns.");
        } finally {
            btnBatchProcess.disabled = false;
            btnBatchProcess.innerHTML = `<i data-lucide="play"></i> Process Batch`;
            refreshIcons();
        }
    });

    function renderBatchResults(rows) {
        batchResults.classList.remove("hidden");

        const total = rows.length;
        const flagged = rows.filter(r => r.is_toxic === 1).length;
        const clean = total - flagged;

        batchTotal.textContent = total;
        batchFlagged.textContent = flagged;
        batchClean.textContent = clean;

        // Render preview table rows
        batchTableBody.innerHTML = "";
        rows.slice(0, 15).forEach(row => {
            const tr = document.createElement("tr");
            
            // Categories flagged
            const categories = categoriesList.filter(c => row[`pred_${c}`] === 1).join(", ");
            const catDisplay = categories ? `<span style="color: var(--danger); font-weight: 600;">${categories.replace(/_/g, ' ')}</span>` : '<span style="color: var(--success);">None</span>';

            tr.innerHTML = `
                <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${row.comment_text}</td>
                <td>${row.detected_language}</td>
                <td><strong>${Math.round(row.severity_score * 100)}%</strong></td>
                <td>${catDisplay}</td>
            `;
            batchTableBody.appendChild(tr);
        });
    }

    btnDownloadReport.addEventListener("click", () => {
        if (!processedCSVContent) return;

        const blob = new Blob([processedCSVContent], { type: "text/csv;charset=utf-8;" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.setAttribute("download", `inditox_moderation_report_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
