/**
 * SAM3 Two-Mouse Tracking widget
 *
 * Adds to SAM3TwoMouseTracking node:
 *  - Prompt list counter ("Prompts: N")
 *  - "Clear List" button → resets prompt_store to "[]"
 *
 * The prompt_store string widget is hidden visually but kept serialized
 * so its value persists in the workflow JSON across ComfyUI restarts.
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SAM3.TwoMouseTracking",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SAM3TwoMouseTracking") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // ── Find and hide prompt_store widget ──────────────────────────
            const storeWidget = this.widgets?.find(w => w.name === "prompt_store");
            if (storeWidget) {
                storeWidget.origType       = storeWidget.type;
                storeWidget.computeSize    = () => [0, -4];
                storeWidget.type           = "converted-widget";
                storeWidget.hidden         = true;
                if (storeWidget.element) {
                    storeWidget.element.style.display    = "none";
                    storeWidget.element.style.visibility = "hidden";
                }
            }

            // ── Info bar: counter + clear button ──────────────────────────
            const bar = document.createElement("div");
            bar.style.cssText = "display:flex;justify-content:space-between;align-items:center;"
                              + "padding:6px 8px;background:#1a1a1a;border-radius:4px;"
                              + "margin:4px 0;box-sizing:border-box;width:100%;";

            const counter = document.createElement("span");
            counter.style.cssText = "color:#aaa;font-size:12px;font-family:monospace;";
            counter.textContent   = "Prompts: 0";
            bar.appendChild(counter);

            const listBtn = document.createElement("button");
            listBtn.textContent  = "List Prompts";
            listBtn.style.cssText = "padding:4px 10px;background:#555;color:#fff;"
                                  + "border:1px solid #333;border-radius:3px;"
                                  + "cursor:pointer;font-size:12px;margin-right:4px;";
            listBtn.onmouseover = () => listBtn.style.background = "#666";
            listBtn.onmouseout  = () => listBtn.style.background = "#555";
            bar.appendChild(listBtn);

            const clearBtn = document.createElement("button");
            clearBtn.textContent  = "Clear List";
            clearBtn.style.cssText = "padding:4px 10px;background:#c44;color:#fff;"
                                   + "border:1px solid #a22;border-radius:3px;"
                                   + "cursor:pointer;font-size:12px;font-weight:bold;";
            clearBtn.onmouseover = () => clearBtn.style.background = "#d55";
            clearBtn.onmouseout  = () => clearBtn.style.background = "#c44";
            bar.appendChild(clearBtn);

            // Add as DOM widget (zero-height if no content)
            const domWidget = this.addDOMWidget("info_bar", "infoBar", bar);
            domWidget.computeSize = (width) => [width, 36];

            this._promptCounter = counter;
            this._storeWidget   = storeWidget;

            // List Prompts button: print to server terminal
            listBtn.addEventListener("click", async (e) => {
                e.preventDefault();
                e.stopPropagation();
                const store = this._storeWidget?.value || "[]";
                await fetch("/sam3/list_prompts", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({prompt_store: store}),
                });
            });

            // Clear button: reset prompt_store to "[]"
            clearBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (this._storeWidget) {
                    this._storeWidget.value = "[]";
                }
                this._promptCounter.textContent = "Prompts: 0";
            });

            // On execution: sync counter, update store, clear upstream collectors
            this.onExecuted = (message) => {
                if (message.prompt_store && message.prompt_store[0] !== undefined) {
                    const newStore = message.prompt_store[0];
                    if (this._storeWidget) {
                        this._storeWidget.value = newStore;
                    }
                }
                if (message.prompt_count && message.prompt_count[0] !== undefined) {
                    this._promptCounter.textContent = `Prompts: ${message.prompt_count[0]}`;
                }

                // Clear all directly connected SAM3PointCollector nodes
                for (const input of (this.inputs || [])) {
                    if (!input.link) continue;
                    const linkInfo = app.graph.links[input.link];
                    if (!linkInfo) continue;
                    const upstream = app.graph.getNodeById(linkInfo.origin_id);
                    if (upstream && upstream.type === "SAM3PointCollector") {
                        upstream.clearCanvas?.();
                    }
                }
            };

            return result;
        };
    },
});
