/**
 * SAM3 Working Dir Selector widget
 *
 * Replaces selected_folder / selected_video text widgets with dropdowns.
 * Queries /sam3/wd/list_folders and /sam3/wd/list_videos from the backend.
 * Shows computed video_path / video_id / csv_path as a read-only info panel.
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

            // Hide selected_folder and selected_video text widgets
            setTimeout(() => {
                for (const name of ["selected_folder", "selected_video"]) {
                    const w = getW(name);
                    if (!w) continue;
                    w.draw = () => {};
                    try {
                        Object.defineProperty(w, "computedHeight", {
                            get: () => 0, set: () => {}, configurable: true,
                        });
                    } catch (e) { w.computedHeight = 0; }
                }
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }, 0);

            // ── DOM container ──────────────────────────────────────────────
            const container = document.createElement("div");
            container.style.cssText =
                "display:flex;flex-direction:column;gap:3px;" +
                "padding:4px 6px;box-sizing:border-box;width:100%;";

            // Row helpers
            const makeRow = () => {
                const r = document.createElement("div");
                r.style.cssText = "display:flex;align-items:center;gap:4px;";
                return r;
            };
            const makeLabel = text => {
                const l = document.createElement("span");
                l.style.cssText =
                    "color:#999;font-size:10px;white-space:nowrap;width:48px;text-align:right;flex-shrink:0;";
                l.textContent = text;
                return l;
            };
            const makeSelect = () => {
                const s = document.createElement("select");
                s.style.cssText =
                    "flex:1;background:#2a2a2a;color:#ddd;border:1px solid #555;" +
                    "border-radius:3px;padding:1px 4px;font-size:11px;min-width:0;";
                return s;
            };

            // ── Folder row ─────────────────────────────────────────────────
            const folderRow = makeRow();
            folderRow.appendChild(makeLabel("Folder:"));

            const folderSel = makeSelect();
            folderRow.appendChild(folderSel);

            const refreshBtn = document.createElement("button");
            refreshBtn.textContent = "🔄";
            refreshBtn.title = "Refresh (re-scan working_dir/video/)";
            refreshBtn.style.cssText =
                "padding:1px 6px;background:#484848;color:#ddd;" +
                "border:1px solid #666;border-radius:3px;cursor:pointer;" +
                "font-size:11px;flex-shrink:0;";
            refreshBtn.onmouseover = () => refreshBtn.style.background = "#585858";
            refreshBtn.onmouseout  = () => refreshBtn.style.background = "#484848";
            folderRow.appendChild(refreshBtn);

            // ── Video row ──────────────────────────────────────────────────
            const videoRow = makeRow();
            videoRow.appendChild(makeLabel("Video:"));

            const videoSel = makeSelect();
            videoRow.appendChild(videoSel);

            container.appendChild(folderRow);
            container.appendChild(videoRow);

            // ── Logic ──────────────────────────────────────────────────────
            const updateInfo = () => {
                const wdW = getW("working_dir");
                const wd = wdW?.value?.trim() || "";
                const folder = folderSel.value;
                const video  = videoSel.value;

                // Sync hidden widgets so they are saved in the workflow JSON
                const fw = getW("selected_folder");
                const vw = getW("selected_video");
                if (fw) fw.value = folder;
                if (vw) vw.value = video;

                if (!folder || !video) return;
            };

            const populateSelect = (sel, items, preserve) => {
                sel.innerHTML = "";
                const blank = document.createElement("option");
                blank.value = "";
                blank.textContent = `— select ${sel === folderSel ? "folder" : "video"} —`;
                sel.appendChild(blank);
                for (const item of items) {
                    const opt = document.createElement("option");
                    opt.value = item;
                    opt.textContent = item;
                    sel.appendChild(opt);
                }
                if (preserve && items.includes(preserve)) sel.value = preserve;
            };

            const loadVideos = async (restoreVideo) => {
                const wdW = getW("working_dir");
                const wd  = wdW?.value?.trim() || "";
                const folder = folderSel.value;
                if (!wd || !folder) {
                    populateSelect(videoSel, [], "");
                    updateInfo();
                    return;
                }
                try {
                    const resp = await fetch(
                        `/sam3/wd/list_videos?working_dir=${encodeURIComponent(wd)}&folder=${encodeURIComponent(folder)}`
                    );
                    const data = await resp.json();
                    populateSelect(videoSel, data.videos || [], restoreVideo);
                } catch (e) {
                    console.error("[SAM3WorkingDir] list_videos:", e);
                }
                updateInfo();
            };

            const loadFolders = async (restoreFolder, restoreVideo) => {
                const wdW = getW("working_dir");
                const wd  = wdW?.value?.trim() || "";
                if (!wd) return;
                try {
                    const resp = await fetch(
                        `/sam3/wd/list_folders?working_dir=${encodeURIComponent(wd)}`
                    );
                    const data = await resp.json();
                    populateSelect(folderSel, data.folders || [], restoreFolder);
                } catch (e) {
                    console.error("[SAM3WorkingDir] list_folders:", e);
                }
                await loadVideos(restoreVideo);
            };

            // Events
            refreshBtn.addEventListener("click", async e => {
                e.preventDefault(); e.stopPropagation();
                await loadFolders(folderSel.value, videoSel.value);
            });

            folderSel.addEventListener("change", () => loadVideos());
            videoSel.addEventListener("change", () => updateInfo());

            // Register DOM widget (height: 2 rows × 26 + gaps)
            const dw = node.addDOMWidget("wd_selector", "wdSelector", container);
            dw.computeSize = w => [w, 62];

            // On load: restore saved selections
            setTimeout(async () => {
                const wdW = getW("working_dir");
                if (!wdW?.value?.trim()) return;
                const savedFolder = getW("selected_folder")?.value || "";
                const savedVideo  = getW("selected_video")?.value  || "";
                await loadFolders(savedFolder, savedVideo);
            }, 300);

            return result;
        };
    },
});
