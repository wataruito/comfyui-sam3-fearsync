/**
 * SAM3 Load Propagation widget
 * Adds a "📂 Browse" button to SAM3LoadPropagation for picking .pt files.
 */

import { app } from "../../scripts/app.js";
import { openFileBrowser } from "./sam3_file_browser.js";

app.registerExtension({
    name: "Comfy.SAM3.LoadPropagation",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SAM3LoadPropagation") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // Hide only save_path text input — DOM browse row replaces it
            // folder_path stays visible so it can be connected via right-click → Convert to input
            setTimeout(() => {
                const pw = this.widgets?.find(w => w.name === "save_path");
                if (pw) {
                    pw.draw = () => {};
                    try {
                        Object.defineProperty(pw, "computedHeight", {
                            get: () => 0, set: () => {}, configurable: true,
                        });
                    } catch (e) { pw.computedHeight = 0; }
                    this.setSize(this.computeSize());
                    this.setDirtyCanvas(true, true);
                }
            }, 0);

            const row = document.createElement("div");
            row.style.cssText =
                "display:flex;align-items:center;gap:4px;" +
                "padding:2px 6px;box-sizing:border-box;width:100%;";

            // Label: shows currently selected filename
            const label = document.createElement("span");
            label.style.cssText =
                "flex:1;background:#1a1a1a;color:#666;border:1px solid #383838;" +
                "border-radius:3px;padding:2px 6px;font-size:10px;" +
                "font-family:monospace;overflow:hidden;text-overflow:ellipsis;" +
                "white-space:nowrap;min-width:0;cursor:default;";
            label.textContent = "— no file selected —";
            row.appendChild(label);

            // Browse button
            const browseB = document.createElement("button");
            browseB.textContent = "📂 Browse";
            browseB.style.cssText =
                "padding:2px 8px;background:#484848;color:#ddd;" +
                "border:1px solid #666;border-radius:3px;cursor:pointer;" +
                "font-size:11px;white-space:nowrap;flex-shrink:0;";
            browseB.onmouseover = () => browseB.style.background = "#585858";
            browseB.onmouseout  = () => browseB.style.background = "#484848";
            row.appendChild(browseB);

            const updateLabel = () => {
                const pw = this.widgets?.find(w => w.name === "save_path");
                const v = pw?.value || "";
                label.textContent = v ? v.split("/").pop() : "— no file selected —";
                label.title       = v;
                label.style.color = v ? "#bbb" : "#666";
            };

            browseB.addEventListener("click", e => {
                e.preventDefault(); e.stopPropagation();
                // folder_path (from SAM3OutputFolder) as initial dir; fall back to save_path dir
                const fpw = this.widgets?.find(w => w.name === "folder_path");
                const pw  = this.widgets?.find(w => w.name === "save_path");
                const folder = fpw?.value?.trim() || "";
                const cur    = pw?.value?.trim() || "";
                const initialPath = folder ||
                    (cur.endsWith(".pt") ? cur.substring(0, cur.lastIndexOf("/")) || null : cur) ||
                    null;
                openFileBrowser({
                    node:           this,
                    widgetName:     "save_path",
                    browseEndpoint: "/sam3/browse_files",
                    fileTypeLabel:  "PyTorch",
                    initialPath,
                    onPicked:       updateLabel,
                });
            });

            const dw = this.addDOMWidget("prop_browse", "propBrowse", row);
            dw.computeSize = w => [w, 26];
            setTimeout(updateLabel, 100);
            return result;
        };
    },
});
