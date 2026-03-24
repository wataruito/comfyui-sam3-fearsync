/**
 * SAM3 Output Folder widget
 * Adds a "📂 Browse" button to SAM3OutputFolder for choosing an output directory.
 */

import { app } from "../../scripts/app.js";
import { openDirBrowser } from "./sam3_file_browser.js";

app.registerExtension({
    name: "Comfy.SAM3.OutputFolder",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SAM3OutputFolder") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const row = document.createElement("div");
            row.style.cssText =
                "display:flex;align-items:center;gap:4px;" +
                "padding:2px 6px;box-sizing:border-box;width:100%;";

            // Label: shows currently selected folder
            const label = document.createElement("span");
            label.style.cssText =
                "flex:1;background:#1a1a1a;color:#666;border:1px solid #383838;" +
                "border-radius:3px;padding:2px 6px;font-size:10px;" +
                "font-family:monospace;overflow:hidden;text-overflow:ellipsis;" +
                "white-space:nowrap;min-width:0;cursor:default;";
            label.textContent = "— no folder selected —";
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
                const pw = this.widgets?.find(w => w.name === "folder");
                const v = pw?.value || "";
                label.textContent = v || "— no folder selected —";
                label.title       = v;
                label.style.color = v ? "#bbb" : "#666";
            };

            browseB.addEventListener("click", e => {
                e.preventDefault(); e.stopPropagation();
                openDirBrowser({
                    node:       this,
                    widgetName: "folder",
                    onPicked:   updateLabel,
                });
            });

            const dw = this.addDOMWidget("folder_browse", "folderBrowse", row);
            dw.computeSize = w => [w, 26];
            setTimeout(updateLabel, 100);
            return result;
        };
    },
});
