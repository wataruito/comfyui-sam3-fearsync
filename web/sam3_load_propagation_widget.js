/**
 * SAM3 Load Propagation widget
 * Adds a file selector to SAM3LoadPropagation for picking .pt files
 * from output/sam3_propagation/.
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SAM3.LoadPropagation",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SAM3LoadPropagation") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const row = document.createElement("div");
            row.style.cssText = "display:flex;align-items:center;gap:4px;"
                              + "padding:2px 6px;box-sizing:border-box;width:100%;";

            const sel = document.createElement("select");
            sel.style.cssText = "flex:1;background:#2a2a2a;color:#ddd;border:1px solid #555;"
                              + "border-radius:3px;padding:2px 4px;font-size:11px;"
                              + "font-family:monospace;min-width:0;";
            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = "— click 🔄 to load .pt files —";
            sel.appendChild(placeholder);
            row.appendChild(sel);

            const refreshBtn = document.createElement("button");
            refreshBtn.textContent = "🔄";
            refreshBtn.title = "Refresh propagation file list";
            refreshBtn.style.cssText = "padding:2px 6px;background:#444;color:#fff;"
                                     + "border:1px solid #666;border-radius:3px;cursor:pointer;font-size:13px;";
            refreshBtn.onmouseover = () => refreshBtn.style.background = "#555";
            refreshBtn.onmouseout  = () => refreshBtn.style.background = "#444";
            row.appendChild(refreshBtn);

            const loadFiles = async () => {
                refreshBtn.textContent = "⏳";
                try {
                    const resp  = await fetch("/sam3/list_propagations");
                    const files = await resp.json();
                    const pathWidget = this.widgets?.find(w => w.name === "save_path");
                    const current    = pathWidget?.value || "";
                    sel.innerHTML    = "";
                    const ph = document.createElement("option");
                    ph.value = ""; ph.textContent = "— select a .pt file —";
                    sel.appendChild(ph);
                    files.forEach(f => {
                        const opt = document.createElement("option");
                        opt.value = f.path;
                        opt.textContent = f.label;
                        if (f.path === current) opt.selected = true;
                        sel.appendChild(opt);
                    });
                } catch (e) {
                    const opt = document.createElement("option");
                    opt.value = ""; opt.textContent = "Error loading list";
                    sel.innerHTML = ""; sel.appendChild(opt);
                }
                refreshBtn.textContent = "🔄";
            };

            refreshBtn.addEventListener("click", (e) => {
                e.preventDefault(); e.stopPropagation();
                loadFiles();
            });

            sel.addEventListener("change", () => {
                if (!sel.value) return;
                const pathWidget = this.widgets?.find(w => w.name === "save_path");
                if (pathWidget) {
                    pathWidget.value = sel.value;
                    const el = pathWidget.element || pathWidget.inputEl;
                    if (el) el.value = sel.value;
                }
            });

            const domWidget = this.addDOMWidget("prop_browse", "propBrowse", row);
            domWidget.computeSize = (width) => [width, 26];

            // Auto-load on first render
            setTimeout(loadFiles, 300);

            return result;
        };
    },
});
