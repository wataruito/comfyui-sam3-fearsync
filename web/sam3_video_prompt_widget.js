/**
 * SAM3 Video Prompt Point widget
 * Interactive canvas for SAM3VideoPromptPoint node.
 * Same click logic as SAM3PointCollector but registered for SAM3VideoPromptPoint.
 *
 * Left-click          → positive point (green)
 * Shift / Right-click → negative point (red)
 * Click existing point with right-button → delete
 */

import { app } from "../../scripts/app.js";

function hideWidgetForGood(node, widget, suffix = '') {
    if (!widget) return;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget" + suffix;
    widget.hidden = true;
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.visibility = "hidden";
    }
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            hideWidgetForGood(node, w, ':' + widget.name);
        }
    }
}

app.registerExtension({
    name: "Comfy.SAM3.VideoPromptPoint",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SAM3VideoPromptPoint") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // ── Canvas container ──────────────────────────────────────────
            const container = document.createElement("div");
            container.style.cssText = "position:relative;width:100%;background:#222;overflow:hidden;box-sizing:border-box;margin:0;padding:0;display:flex;align-items:center;justify-content:center;";

            const infoBar = document.createElement("div");
            infoBar.style.cssText = "position:absolute;top:5px;left:5px;right:5px;z-index:10;display:flex;justify-content:space-between;align-items:center;";
            container.appendChild(infoBar);

            const pointsCounter = document.createElement("div");
            pointsCounter.style.cssText = "padding:5px 10px;background:rgba(0,0,0,0.7);color:#fff;border-radius:3px;font-size:12px;font-family:monospace;";
            pointsCounter.textContent = "Points: 0 pos, 0 neg";
            infoBar.appendChild(pointsCounter);

            const clearButton = document.createElement("button");
            clearButton.textContent = "Clear All";
            clearButton.style.cssText = "padding:5px 10px;background:#d44;color:#fff;border:1px solid #a22;border-radius:3px;cursor:pointer;font-size:12px;font-weight:bold;";
            clearButton.onmouseover = () => clearButton.style.background = "#e55";
            clearButton.onmouseout  = () => clearButton.style.background = "#d44";
            infoBar.appendChild(clearButton);

            const canvas = document.createElement("canvas");
            canvas.width  = 400;
            canvas.height = 300;
            canvas.style.cssText = "display:block;max-width:100%;max-height:100%;object-fit:contain;cursor:crosshair;margin:0 auto;";
            container.appendChild(canvas);

            const ctx = canvas.getContext("2d");

            this.canvasWidget = {
                canvas, ctx, container,
                image: null,
                positivePoints: [],
                negativePoints: [],
                hoveredPoint: null,
                pointsCounter,
                widgetHeight: 300,
            };

            const domWidget = this.addDOMWidget("canvas", "customCanvas", container);
            this.canvasWidget.domWidget = domWidget;
            domWidget.computeSize = (width) => [width, this.canvasWidget.widgetHeight];

            // ── Hide storage widgets ──────────────────────────────────────
            const coordsWidget    = this.widgets.find(w => w.name === "coordinates");
            const negCoordsWidget = this.widgets.find(w => w.name === "neg_coordinates");
            const storeWidget     = this.widgets.find(w => w.name === "points_store");

            if (coordsWidget)    { coordsWidget.value    = coordsWidget.value    || "[]"; }
            if (negCoordsWidget) { negCoordsWidget.value = negCoordsWidget.value || "[]"; }
            if (storeWidget)     { storeWidget.value     = storeWidget.value     || "{}"; }

            this._hiddenWidgets = {
                coordinates:     coordsWidget,
                neg_coordinates: negCoordsWidget,
                points_store:    storeWidget,
            };

            if (coordsWidget)    hideWidgetForGood(this, coordsWidget);
            if (negCoordsWidget) hideWidgetForGood(this, negCoordsWidget);
            if (storeWidget)     hideWidgetForGood(this, storeWidget);

            // Prevent hidden widgets from rendering
            const origDraw = this.onDrawForeground;
            this.onDrawForeground = function(ctx) {
                const hw = this.widgets.filter(w => w.type?.includes("converted-widget"));
                const ot = hw.map(w => w.type);
                hw.forEach(w => w.type = null);
                if (origDraw) origDraw.apply(this, arguments);
                hw.forEach((w, i) => w.type = ot[i]);
            };

            // ── Event handlers ────────────────────────────────────────────
            clearButton.addEventListener("click", (e) => {
                e.preventDefault(); e.stopPropagation();
                this.canvasWidget.positivePoints = [];
                this.canvasWidget.negativePoints = [];
                this.updatePoints();
                this.redrawCanvas();
            });

            canvas.addEventListener("click", (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width)  * canvas.width;
                const y = ((e.clientY - rect.top)  / rect.height) * canvas.height;

                const hit = this.findPointAt(x, y);
                if (hit && e.button === 2) {
                    if (hit.type === 'positive') {
                        this.canvasWidget.positivePoints = this.canvasWidget.positivePoints.filter(p => p !== hit.point);
                    } else {
                        this.canvasWidget.negativePoints = this.canvasWidget.negativePoints.filter(p => p !== hit.point);
                    }
                } else {
                    if (e.shiftKey || e.button === 2) {
                        this.canvasWidget.negativePoints.push({x, y});
                    } else {
                        this.canvasWidget.positivePoints.push({x, y});
                    }
                }
                this.updatePoints();
                this.redrawCanvas();
            });

            canvas.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                canvas.dispatchEvent(new MouseEvent('click', {
                    button: 2, clientX: e.clientX, clientY: e.clientY, shiftKey: e.shiftKey
                }));
            });

            canvas.addEventListener("mousemove", (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width)  * canvas.width;
                const y = ((e.clientY - rect.top)  / rect.height) * canvas.height;
                const hovered = this.findPointAt(x, y);
                if (hovered !== this.canvasWidget.hoveredPoint) {
                    this.canvasWidget.hoveredPoint = hovered;
                    this.redrawCanvas();
                }
            });

            // ── Image update from backend ─────────────────────────────────
            this.onExecuted = (message) => {
                if (message.bg_image && message.bg_image[0]) {
                    const img = new Image();
                    img.onload = () => {
                        this.canvasWidget.image = img;
                        canvas.width  = img.width;
                        canvas.height = img.height;

                        const nodeWidth   = this.size[0] || 400;
                        const aspectRatio = img.height / img.width;
                        const newHeight   = Math.round((nodeWidth - 20) * aspectRatio);

                        this._isResizing = true;
                        this.canvasWidget.widgetHeight = newHeight;
                        container.style.height = newHeight + "px";
                        this.setSize([nodeWidth, newHeight + 80]);
                        setTimeout(() => { this._isResizing = false; }, 50);

                        this.redrawCanvas();
                    };
                    img.src = "data:image/jpeg;base64," + message.bg_image[0];
                }
            };

            const origOnResize = this.onResize;
            this.onResize = function(size) {
                if (origOnResize) origOnResize.apply(this, arguments);
                if (this._isResizing) return;
                this._isResizing = true;
                const newH = Math.max(200, size[1] - 80);
                if (Math.abs(newH - this.canvasWidget.widgetHeight) > 5) {
                    this.canvasWidget.widgetHeight = newH;
                    container.style.height = newH + "px";
                }
                setTimeout(() => { this._isResizing = false; }, 50);
            };

            this.redrawCanvas();
            this.setSize([Math.max(400, this.size[0] || 400), 380]);
            container.style.height = "300px";

            return result;
        };

        // ── Helpers ───────────────────────────────────────────────────────

        nodeType.prototype.findPointAt = function(x, y) {
            const t = 10;
            for (const p of this.canvasWidget.positivePoints)
                if (Math.abs(p.x - x) < t && Math.abs(p.y - y) < t)
                    return {type: 'positive', point: p};
            for (const p of this.canvasWidget.negativePoints)
                if (Math.abs(p.x - x) < t && Math.abs(p.y - y) < t)
                    return {type: 'negative', point: p};
            return null;
        };

        nodeType.prototype.updatePoints = function() {
            const cw  = this._hiddenWidgets?.coordinates     || this.widgets.find(w => w.name === "coordinates");
            const ncw = this._hiddenWidgets?.neg_coordinates || this.widgets.find(w => w.name === "neg_coordinates");
            const sw  = this._hiddenWidgets?.points_store    || this.widgets.find(w => w.name === "points_store");

            if (cw)  cw.value  = JSON.stringify(this.canvasWidget.positivePoints);
            if (ncw) ncw.value = JSON.stringify(this.canvasWidget.negativePoints);
            if (sw)  sw.value  = JSON.stringify({
                positive: this.canvasWidget.positivePoints,
                negative: this.canvasWidget.negativePoints,
            });

            const pos = this.canvasWidget.positivePoints.length;
            const neg = this.canvasWidget.negativePoints.length;
            this.canvasWidget.pointsCounter.textContent = `Points: ${pos} pos, ${neg} neg`;
        };

        nodeType.prototype.redrawCanvas = function() {
            const {canvas, ctx, image, positivePoints, negativePoints, hoveredPoint} = this.canvasWidget;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (image) {
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                // Image-size overlay
                ctx.fillStyle = "rgba(0,0,0,0.7)";
                ctx.fillRect(5, canvas.height - 25, 160, 20);
                ctx.fillStyle = "#0f0";
                ctx.font = "12px monospace";
                ctx.textAlign = "left";
                ctx.fillText(`Image: ${canvas.width}x${canvas.height}`, 10, canvas.height - 10);
            } else {
                ctx.fillStyle = "#333";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#666";
                ctx.font = "16px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Connect image & run to load frame", canvas.width/2, canvas.height/2 - 10);
                ctx.fillText("Left-click: Positive (green)",      canvas.width/2, canvas.height/2 + 20);
                ctx.fillText("Shift/Right: Negative (red)",       canvas.width/2, canvas.height/2 + 45);
            }

            // Positive points — green
            for (const p of positivePoints) {
                const hov = hoveredPoint?.point === p;
                ctx.beginPath();
                ctx.arc(p.x, p.y, hov ? 8 : 6, 0, 2*Math.PI);
                ctx.fillStyle = "#0f0"; ctx.fill();
                ctx.strokeStyle = "#0f0"; ctx.lineWidth = 2; ctx.stroke();
            }
            // Negative points — red
            for (const p of negativePoints) {
                const hov = hoveredPoint?.point === p;
                ctx.beginPath();
                ctx.arc(p.x, p.y, hov ? 8 : 6, 0, 2*Math.PI);
                ctx.fillStyle = "#f00"; ctx.fill();
                ctx.strokeStyle = "#f00"; ctx.lineWidth = 2; ctx.stroke();
            }
        };
    }
});
