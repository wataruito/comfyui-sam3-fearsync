/**
 * SAM3 Working Dir Selector widget
 *
 * Reads pipeline.csv from working_dir and shows video_id values in a dropdown.
 * Status is shown alongside each video_id (e.g. "20250718_f402cd  [new]").
 * The annotator selects a video_id; the node resolves video_path server-side.
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SAM3.WorkingDirSelector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SAM3WorkingDirSelector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            const getW = name => node.widgets?.find(w => w.name === name);

            // Hide selected_video_id text widget
            setTimeout(() => {
                const w = getW("selected_video_id");
                if (!w) return;
                w.draw = () => {};
                try {
                    Object.defineProperty(w, "computedHeight", {
                        get: () => 0, set: () => {}, configurable: true,
                    });
                } catch (e) { w.computedHeight = 0; }
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }, 0);

            // ── DOM container ──────────────────────────────────────────────
            const container = document.createElement("div");
            container.style.cssText =
                "display:flex;flex-direction:column;gap:3px;" +
                "padding:4px 6px;box-sizing:border-box;width:100%;";

            const makeRow = () => {
                const r = document.createElement("div");
                r.style.cssText = "display:flex;align-items:center;gap:4px;";
                return r;
            };
            const makeLabel = text => {
                const l = document.createElement("span");
                l.style.cssText =
                    "color:#999;font-size:10px;white-space:nowrap;width:48px;" +
                    "text-align:right;flex-shrink:0;";
                l.textContent = text;
                return l;
            };

            // ── Video row ──────────────────────────────────────────────────
            const videoRow = makeRow();
            videoRow.appendChild(makeLabel("Video:"));

            const videoSel = document.createElement("select");
            videoSel.style.cssText =
                "flex:1;background:#2a2a2a;color:#ddd;border:1px solid #555;" +
                "border-radius:3px;padding:1px 4px;font-size:11px;min-width:0;";
            videoRow.appendChild(videoSel);

            const refreshBtn = document.createElement("button");
            refreshBtn.textContent = "🔄";
            refreshBtn.title = "Refresh (re-read pipeline.csv)";
            refreshBtn.style.cssText =
                "padding:1px 6px;background:#484848;color:#ddd;" +
                "border:1px solid #666;border-radius:3px;cursor:pointer;" +
                "font-size:11px;flex-shrink:0;";
            refreshBtn.onmouseover = () => refreshBtn.style.background = "#585858";
            refreshBtn.onmouseout  = () => refreshBtn.style.background = "#484848";
            videoRow.appendChild(refreshBtn);

            container.appendChild(videoRow);

            // ── Status color map ───────────────────────────────────────────
            const STATUS_COLOR = {
                "new":       "#888",
                "prompted":  "#4af",
                "tracked":   "#8f8",
                "corrected": "#fa4",
                "done":      "#aaa",
            };

            // ── Logic ──────────────────────────────────────────────────────
            const updateHidden = () => {
                const w = getW("selected_video_id");
                if (w) w.value = videoSel.value;
            };

            const populateSelect = (videos, preserve) => {
                videoSel.innerHTML = "";
                const blank = document.createElement("option");
                blank.value = "";
                blank.textContent = "— select video —";
                videoSel.appendChild(blank);
                for (const { video_id, status } of videos) {
                    const opt = document.createElement("option");
                    opt.value = video_id;
                    opt.textContent = `${video_id}  [${status}]`;
                    opt.style.color = STATUS_COLOR[status] || "#ddd";
                    videoSel.appendChild(opt);
                }
                if (preserve && videos.some(v => v.video_id === preserve)) {
                    videoSel.value = preserve;
                }
                updateHidden();
            };

            const loadVideos = async (restore) => {
                const wdW = getW("working_dir");
                const wd  = wdW?.value?.trim() || "";
                if (!wd) { populateSelect([], ""); return; }
                try {
                    const resp = await fetch(
                        `/sam3/wd/list_pipeline_videos?working_dir=${encodeURIComponent(wd)}`
                    );
                    const data = await resp.json();
                    populateSelect(data.videos || [], restore);
                } catch (e) {
                    console.error("[SAM3WorkingDir] list_pipeline_videos:", e);
                }
            };

            // Events
            refreshBtn.addEventListener("click", async e => {
                e.preventDefault(); e.stopPropagation();
                await loadVideos(videoSel.value);
            });
            videoSel.addEventListener("change", () => updateHidden());

            // Register DOM widget (single row)
            const dw = node.addDOMWidget("wd_selector", "wdSelector", container);
            dw.computeSize = w => [w, 36];

            // On load: restore saved selection
            setTimeout(async () => {
                const wdW = getW("working_dir");
                if (!wdW?.value?.trim()) return;
                const savedId = getW("selected_video_id")?.value || "";
                await loadVideos(savedId);
            }, 300);

            return result;
        };
    },
});
