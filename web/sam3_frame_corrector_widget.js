/**
 * SAM3 Frame Corrector widget
 *
 * Adds to SAM3FrameCorrector node:
 *  - Correction count label ("Corrections: N")
 *  - "List Corrections" button → prints to server terminal
 *  - "Clear Corrections" button → resets correction_store to "[]"
 *
 * The correction_store string widget is hidden visually but serialized
 * so its value persists in the workflow JSON across ComfyUI restarts.
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SAM3.FrameCorrector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SAM3FrameCorrector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // ── Hide correction_store widget ───────────────────────────────
            const storeWidget = this.widgets?.find(w => w.name === "correction_store");
            if (storeWidget) {
                storeWidget.origType    = storeWidget.type;
                storeWidget.computeSize = () => [0, -4];
                storeWidget.type        = "converted-widget";
                storeWidget.hidden      = true;
                if (storeWidget.element) {
                    storeWidget.element.style.display    = "none";
                    storeWidget.element.style.visibility = "hidden";
                }
            }

            // ── Info bar ───────────────────────────────────────────────────
            const bar = document.createElement("div");
            bar.style.cssText = "display:flex;justify-content:space-between;align-items:center;"
                              + "padding:6px 8px;background:#1a1a2e;border-radius:4px;"
                              + "margin:4px 0;box-sizing:border-box;width:100%;"
                              + "border:1px solid #2a2a4e;";

            const counter = document.createElement("span");
            counter.style.cssText = "color:#8af;font-size:12px;font-family:monospace;";
            counter.textContent   = "Corrections: 0";
            bar.appendChild(counter);

            const listBtn = document.createElement("button");
            listBtn.textContent   = "List";
            listBtn.style.cssText = "padding:4px 10px;background:#335;color:#adf;"
                                  + "border:1px solid #44a;border-radius:3px;"
                                  + "cursor:pointer;font-size:12px;margin-right:4px;";
            listBtn.onmouseover = () => listBtn.style.background = "#446";
            listBtn.onmouseout  = () => listBtn.style.background = "#335";
            bar.appendChild(listBtn);

            const clearBtn = document.createElement("button");
            clearBtn.textContent   = "Clear";
            clearBtn.style.cssText = "padding:4px 10px;background:#c44;color:#fff;"
                                   + "border:1px solid #a22;border-radius:3px;"
                                   + "cursor:pointer;font-size:12px;font-weight:bold;";
            clearBtn.onmouseover = () => clearBtn.style.background = "#d55";
            clearBtn.onmouseout  = () => clearBtn.style.background = "#c44";
            bar.appendChild(clearBtn);

            const domWidget = this.addDOMWidget("corrector_bar", "correctorBar", bar);
            domWidget.computeSize = (width) => [width, 36];

            this._correctionCounter = counter;
            this._correctionStore   = storeWidget;

            // List button
            listBtn.addEventListener("click", async (e) => {
                e.preventDefault();
                e.stopPropagation();
                const store = this._correctionStore?.value || "[]";
                await fetch("/sam3/list_corrections", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ correction_store: store }),
                });
            });

            // Clear button
            clearBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (this._correctionStore) {
                    this._correctionStore.value = "[]";
                }
                this._correctionCounter.textContent = "Corrections: 0";
            });

            // On execution: sync counter + store, clear upstream collector
            this.onExecuted = (message) => {
                if (message.correction_store?.[0] !== undefined) {
                    if (this._correctionStore) {
                        this._correctionStore.value = message.correction_store[0];
                    }
                }
                if (message.correction_count?.[0] !== undefined) {
                    this._correctionCounter.textContent =
                        `Corrections: ${message.correction_count[0]}`;
                }

                // Clear directly connected SAM3PointCollector nodes
                for (const input of (this.inputs || [])) {
                    if (!input.link) continue;
                    const linkInfo = app.graph.links[input.link];
                    if (!linkInfo) continue;
                    const upstream = app.graph.getNodeById(linkInfo.origin_id);
                    if (upstream && (upstream.type === "SAM3PointCollector" || upstream.type === "SAM3AnimalPointCollector")) {
                        upstream.clearCanvas?.();
                    }
                }
            };

            return result;
        };
    },
});
