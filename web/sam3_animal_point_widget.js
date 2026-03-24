/**
 * SAM3 Simple Point Collector
 * Uses plain HTML5 Canvas instead of Protovis for better compatibility
 * Version: 2025-01-20-v8-SERIALIZE-FIX
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { openFileBrowser } from "./sam3_file_browser.js";

console.log("[SAM3] ===== VERSION 8 - SERIALIZATION ENABLED =====");

// VHS resize pattern: fit node height to aspect ratio
function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

// Helper function to properly hide widgets (enhanced for complete hiding)
function hideWidgetForGood(node, widget, suffix = '') {
    if (!widget) return;

    // Save original properties
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;

    // Multiple hiding approaches to ensure widget is fully hidden
    widget.computeSize = () => [0, -4];  // -4 compensates for litegraph's automatic widget gap
    widget.type = "converted-widget" + suffix;
    widget.hidden = true;  // Mark as hidden

    // IMPORTANT: Keep serialization enabled so values are sent to backend
    // (We just hide it visually, but it still needs to send data)

    // Make the widget completely invisible in the DOM if it has element
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.visibility = "hidden";
    }

    // Handle linked widgets recursively
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            hideWidgetForGood(node, w, ':' + widget.name);
        }
    }
}

app.registerExtension({
    name: "Comfy.SAM3.AnimalPointCollector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("[SAM3] beforeRegisterNodeDef called for:", nodeData.name);

        if (nodeData.name === "SAM3AnimalPointCollector") {
            console.log("[SAM3] Registering SAM3AnimalPointCollector node");
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                console.log("[SAM3] onNodeCreated called for SAM3AnimalPointCollector");

                // Call original onNodeCreated FIRST to create all widgets
                const result = onNodeCreated?.apply(this, arguments);

                console.log("[SAM3] Widgets after creation:", this.widgets?.map(w => w.name));

                console.log("[SAM3] Creating canvas container");
                // Create canvas container - dynamically sized based on node height
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; height: 100%; background: #222; overflow: hidden; box-sizing: border-box; margin: 0; padding: 0; display: flex; align-items: center; justify-content: center;";

                // Create info/button bar
                const infoBar = document.createElement("div");
                infoBar.style.cssText = "position: absolute; top: 5px; left: 5px; right: 5px; z-index: 10; display: flex; justify-content: space-between; align-items: center;";
                container.appendChild(infoBar);

                // Create points counter
                const pointsCounter = document.createElement("div");
                pointsCounter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 12px; font-family: monospace;";
                pointsCounter.textContent = "Points: 0 pos, 0 neg";
                infoBar.appendChild(pointsCounter);

                // Create mask toggle button
                const maskToggle = document.createElement("button");
                maskToggle.textContent = "Mask: OFF";
                maskToggle.style.cssText = "padding: 5px 10px; background: #444; color: #aaa; border: 1px solid #666; border-radius: 3px; cursor: pointer; font-size: 12px; font-weight: bold; display: none;";
                maskToggle.addEventListener("click", (e) => {
                    e.preventDefault(); e.stopPropagation();
                    this.canvasWidget.showMask = !this.canvasWidget.showMask;
                    if (this.canvasWidget.showMask) {
                        maskToggle.textContent = "Mask: ON";
                        maskToggle.style.background = "#484";
                        maskToggle.style.color = "#fff";
                        maskToggle.style.borderColor = "#4a4";
                        // Fetch mask frame for current position
                        const idx = parseInt(this._videoControls?.slider?.value || 0);
                        this._fetchMaskFrame(idx);
                    } else {
                        maskToggle.textContent = "Mask: OFF";
                        maskToggle.style.background = "#444";
                        maskToggle.style.color = "#aaa";
                        maskToggle.style.borderColor = "#666";
                        this.canvasWidget.maskImage = null;
                        this.redrawCanvas();
                    }
                });
                infoBar.appendChild(maskToggle);
                this._maskToggleBtn = maskToggle;

                // Create clear button
                const clearButton = document.createElement("button");
                clearButton.textContent = "Clear All";
                clearButton.style.cssText = "padding: 5px 10px; background: #d44; color: #fff; border: 1px solid #a22; border-radius: 3px; cursor: pointer; font-size: 12px; font-weight: bold;";
                clearButton.onmouseover = () => clearButton.style.background = "#e55";
                clearButton.onmouseout = () => clearButton.style.background = "#d44";
                infoBar.appendChild(clearButton);

                // Create canvas for image and points
                const canvas = document.createElement("canvas");
                canvas.width = 400;
                canvas.height = 300;
                // Use max-width and max-height instead of width/height 100% to prevent overflow
                canvas.style.cssText = "display: block; width: 100%; height: 100%; object-fit: contain; cursor: crosshair;";
                container.appendChild(canvas);

                const ctx = canvas.getContext("2d");
                console.log("[SAM3] Canvas created:", canvas);

                // Store state
                this.canvasWidget = {
                    canvas: canvas,
                    ctx: ctx,
                    container: container,
                    image: null,
                    maskImage: null,
                    showMask: false,
                    aspectRatio: null,
                    positivePoints: [],
                    negativePoints: [],
                    hoveredPoint: null,
                    pointsCounter: pointsCounter
                };

                // ── mask_video_path file selector (above canvas) ──────────
                {
                    const row = document.createElement("div");
                    row.style.cssText = "display:flex;align-items:center;gap:4px;"
                                      + "padding:2px 6px;box-sizing:border-box;width:100%;";

                    // Label: shows currently selected filename
                    const label = document.createElement("span");
                    label.style.cssText = "flex:1;background:#1a1a1a;color:#666;border:1px solid #383838;"
                                        + "border-radius:3px;padding:2px 6px;font-size:10px;"
                                        + "font-family:monospace;overflow:hidden;text-overflow:ellipsis;"
                                        + "white-space:nowrap;min-width:0;cursor:default;";
                    label.textContent = "— no mask video —";
                    row.appendChild(label);

                    const browseBtn = document.createElement("button");
                    browseBtn.textContent = "📂 Browse";
                    browseBtn.style.cssText = "padding:2px 8px;background:#484848;color:#ddd;"
                                            + "border:1px solid #666;border-radius:3px;cursor:pointer;"
                                            + "font-size:11px;white-space:nowrap;flex-shrink:0;";
                    browseBtn.onmouseover = () => browseBtn.style.background = "#585858";
                    browseBtn.onmouseout  = () => browseBtn.style.background = "#484848";
                    row.appendChild(browseBtn);

                    const updateMaskLabel = () => {
                        const pw = this.widgets?.find(w => w.name === "mask_video_path");
                        const v = pw?.value || "";
                        label.textContent = v ? v.split("/").pop() : "— no mask video —";
                        label.title       = v;
                        label.style.color = v ? "#bbb" : "#666";
                        // Show mask toggle if a video is set
                        if (this._maskToggleBtn) {
                            this._maskToggleBtn.style.display = v ? "" : "none";
                        }
                    };

                    browseBtn.addEventListener("click", (e) => {
                        e.preventDefault(); e.stopPropagation();
                        openFileBrowser({
                            node:           this,
                            widgetName:     "mask_video_path",
                            browseEndpoint: "/sam3/browse_videos",
                            fileTypeLabel:  "Video",
                            onPicked:       updateMaskLabel,
                        });
                    });

                    const browseWidget = this.addDOMWidget("mask_browse", "maskBrowse", row);
                    browseWidget.computeSize = (width) => [width, 26];

                    setTimeout(updateMaskLabel, 100);
                }
                // ── end file selector ──────────────────────────────────

                // ── Video playback controls (above canvas) ───────────────
                const controls = document.createElement("div");
                controls.style.cssText = "display:flex;align-items:center;gap:5px;padding:4px 6px;"
                                       + "background:#1a1a1a;box-sizing:border-box;width:100%;";

                const playBtn = document.createElement("button");
                playBtn.textContent = "▶";
                playBtn.style.cssText = "padding:2px 8px;background:#444;color:#fff;border:1px solid #555;"
                                      + "border-radius:3px;cursor:pointer;font-size:13px;min-width:30px;";
                playBtn.onmouseover = () => playBtn.style.background = "#555";
                playBtn.onmouseout  = () => playBtn.style.background = "#444";
                controls.appendChild(playBtn);

                const timeLabel = document.createElement("span");
                timeLabel.textContent = "0:00 / 0:00";
                timeLabel.style.cssText = "color:#aaa;font-size:11px;font-family:monospace;min-width:85px;";
                controls.appendChild(timeLabel);

                const slider = document.createElement("input");
                slider.type = "range"; slider.min = 0; slider.max = 0; slider.value = 0;
                slider.style.cssText = "flex:1;cursor:pointer;accent-color:#4af;";
                controls.appendChild(slider);

                const frameLabel = document.createElement("span");
                frameLabel.textContent = "Frame: 0";
                frameLabel.style.cssText = "color:#aaa;font-size:11px;font-family:monospace;"
                                         + "min-width:68px;text-align:right;";
                controls.appendChild(frameLabel);

                const ctrlWidget = this.addDOMWidget("controls", "videoControls", controls);
                ctrlWidget.computeSize = (width) => [width, 30];

                // Add canvas as DOM widget (below file selector and controls).
                console.log("[SAM3] Adding DOM widget via addDOMWidget");
                const widget = this.addDOMWidget("canvas", "customCanvas", container);
                console.log("[SAM3] addDOMWidget returned:", widget);

                // Store widget reference for updates
                this.canvasWidget.domWidget = widget;

                // setCanvasHeight: updates computedHeight (LiteGraph) and container CSS.
                const setCanvasHeight = (h) => {
                    h = Math.max(50, h);
                    widget.computedHeight = h + 10;
                    container.style.height = h + "px";
                };

                widget.computeSize = (width) => {
                    if (this.canvasWidget.aspectRatio) {
                        const h = Math.max(50, (this.size[0] - 20) / this.canvasWidget.aspectRatio);
                        setCanvasHeight(h);
                        return [width, h];
                    }
                    setCanvasHeight(300);
                    return [width, 300];
                };

                // onResize: enforce aspect-ratio height for any resize direction.
                // Modifies size[1] in-place so the window snaps to the video size.
                this.onResize = (size) => {
                    if (!this.canvasWidget.aspectRatio) return;
                    const h = Math.max(50, (size[0] - 20) / this.canvasWidget.aspectRatio);
                    setCanvasHeight(h);
                    size[1] = this.computeSize([size[0], 0])[1];
                };

                // Video state
                this.videoState = {
                    nodeId:      null,
                    totalFrames: 1,
                    fps:         4,
                    isPlaying:   false,
                    playInterval: null,
                };

                const formatTime = (secs) => {
                    const m = Math.floor(secs / 60);
                    const s = Math.floor(secs % 60);
                    return `${m}:${s.toString().padStart(2, "0")}`;
                };

                const updateFrameDisplay = (idx) => {
                    slider.value = idx;
                    frameLabel.textContent = `Frame: ${idx}`;
                    const fps   = this.videoState.fps;
                    const total = this.videoState.totalFrames;
                    timeLabel.textContent = `${formatTime(idx / fps)} / ${formatTime((total - 1) / fps)}`;
                    // Sync frame_idx widget
                    const fw = this.widgets?.find(w => w.name === "frame_idx");
                    if (fw) fw.value = idx;
                };

                const fetchFrame = async (idx) => {
                    if (!this.videoState.nodeId) return;
                    try {
                        const resp = await fetch("/sam3/get_frame", {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({node_id: this.videoState.nodeId, frame_idx: idx}),
                        });
                        if (!resp.ok) return;
                        const blob = await resp.blob();
                        const url  = URL.createObjectURL(blob);
                        const img  = new Image();
                        img.onload = () => {
                            URL.revokeObjectURL(url);
                            this.canvasWidget.image = img;
                            canvas.width  = img.width;
                            canvas.height = img.height;
                            this.canvasWidget.aspectRatio = img.width / img.height;
                            setCanvasHeight(Math.max(50, (this.size[0] - 20) / this.canvasWidget.aspectRatio));
                            fitHeight(this);
                            this.redrawCanvas();
                        };
                        img.src = url;
                    } catch (e) { /* ignore */ }
                };

                // Fetch mask frame on demand
                this._fetchMaskFrame = async (idx) => {
                    if (!this.videoState.nodeId || !this.canvasWidget.showMask) return;
                    try {
                        const resp = await fetch("/sam3/get_mask_frame", {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({node_id: this.videoState.nodeId, frame_idx: idx}),
                        });
                        if (!resp.ok) return;
                        const blob = await resp.blob();
                        const url  = URL.createObjectURL(blob);
                        const img  = new Image();
                        img.onload = () => {
                            URL.revokeObjectURL(url);
                            this.canvasWidget.maskImage = img;
                            this.redrawCanvas();
                        };
                        img.src = url;
                    } catch (e) { /* ignore */ }
                };

                // Slider scrub
                slider.addEventListener("input", () => {
                    const idx = parseInt(slider.value);
                    updateFrameDisplay(idx);
                    fetchFrame(idx);
                    this._fetchMaskFrame(idx);
                });

                // Play / Pause
                playBtn.addEventListener("click", (e) => {
                    e.preventDefault(); e.stopPropagation();
                    if (this.videoState.isPlaying) {
                        clearInterval(this.videoState.playInterval);
                        this.videoState.isPlaying = false;
                        playBtn.textContent = "▶";
                    } else {
                        this.videoState.isPlaying = true;
                        playBtn.textContent = "⏸";
                        this.videoState.playInterval = setInterval(() => {
                            let next = parseInt(slider.value) + 1;
                            if (next >= this.videoState.totalFrames) next = 0;
                            updateFrameDisplay(next);
                            fetchFrame(next);
                            this._fetchMaskFrame(next);
                        }, Math.round(1000 / this.videoState.fps));
                    }
                });

                // Store refs for onExecuted
                this._videoControls = { slider, frameLabel, timeLabel, playBtn, updateFrameDisplay, fetchFrame };

                // Clear button handler
                clearButton.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log("[SAM3] Clearing all points");
                    this.canvasWidget.positivePoints = [];
                    this.canvasWidget.negativePoints = [];
                    this.updatePoints();
                    this.redrawCanvas();
                });

                // Hide the string storage widgets - multiple approaches
                console.log("[SAM3] Attempting to hide widgets...");
                console.log("[SAM3] Widgets before hiding:", this.widgets.map(w => w.name));

                const coordsWidget = this.widgets.find(w => w.name === "coordinates");
                const negCoordsWidget = this.widgets.find(w => w.name === "neg_coordinates");
                const storeWidget = this.widgets.find(w => w.name === "points_store");
                const maskVideoWidget = this.widgets.find(w => w.name === "mask_video_path");

                console.log("[SAM3] Found widgets to hide:", { coordsWidget, negCoordsWidget, storeWidget });

                // Initialize default values BEFORE hiding
                if (coordsWidget) {
                    coordsWidget.value = coordsWidget.value || "[]";
                }
                if (negCoordsWidget) {
                    negCoordsWidget.value = negCoordsWidget.value || "[]";
                }
                if (storeWidget) {
                    storeWidget.value = storeWidget.value || "{}";
                }

                // Store references before hiding
                this._hiddenWidgets = {
                    coordinates: coordsWidget,
                    neg_coordinates: negCoordsWidget,
                    points_store: storeWidget
                };

                // Apply hiding
                if (coordsWidget) {
                    hideWidgetForGood(this, coordsWidget);
                    console.log("[SAM3] coordinates - type:", coordsWidget.type, "hidden:", coordsWidget.hidden, "value:", coordsWidget.value);
                }
                if (negCoordsWidget) {
                    hideWidgetForGood(this, negCoordsWidget);
                    console.log("[SAM3] neg_coordinates - type:", negCoordsWidget.type, "hidden:", negCoordsWidget.hidden, "value:", negCoordsWidget.value);
                }
                if (storeWidget) {
                    hideWidgetForGood(this, storeWidget);
                    console.log("[SAM3] points_store - type:", storeWidget.type, "hidden:", storeWidget.hidden, "value:", storeWidget.value);
                }
                // mask_video_path is NOT hidden — it needs to be visible as an input socket
                // so SAM3OutputFolder.folder_path can be connected to it.

                // CRITICAL FIX: Override onDrawForeground to skip rendering hidden widgets
                const originalDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function(ctx) {
                    // Temporarily hide converted widgets from rendering
                    const hiddenWidgets = this.widgets.filter(w => w.type?.includes("converted-widget"));
                    const originalTypes = hiddenWidgets.map(w => w.type);

                    // Temporarily set to null to prevent rendering
                    hiddenWidgets.forEach(w => w.type = null);

                    // Call original draw
                    if (originalDrawForeground) {
                        originalDrawForeground.apply(this, arguments);
                    }

                    // Restore types
                    hiddenWidgets.forEach((w, i) => w.type = originalTypes[i]);
                };

                console.log("[SAM3] Widgets after hiding:", this.widgets.map(w => `${w.name}(${w.type})`));
                console.log("[SAM3] All widgets processing complete");

                // Mouse event handlers
                canvas.addEventListener("click", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    // Calculate coordinates relative to the actual image dimensions
                    // The canvas might be scaled, so we need to map from display coords to image coords
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                    console.log(`[SAM3] Click at canvas coords: (${x.toFixed(1)}, ${y.toFixed(1)}), canvas size: ${canvas.width}x${canvas.height}`);

                    // Check if clicking existing point to delete
                    const clickedPoint = this.findPointAt(x, y);
                    if (clickedPoint && e.button === 2) {
                        // Right-click on existing point = delete
                        if (clickedPoint.type === 'positive') {
                            this.canvasWidget.positivePoints = this.canvasWidget.positivePoints.filter(p => p !== clickedPoint.point);
                        } else {
                            this.canvasWidget.negativePoints = this.canvasWidget.negativePoints.filter(p => p !== clickedPoint.point);
                        }
                    } else {
                        // Add new point
                        if (e.shiftKey || e.button === 2) {
                            // Negative point
                            this.canvasWidget.negativePoints.push({x, y});
                        } else {
                            // Positive point
                            this.canvasWidget.positivePoints.push({x, y});
                        }
                    }

                    this.updatePoints();
                    this.redrawCanvas();
                });

                canvas.addEventListener("contextmenu", (e) => {
                    e.preventDefault();
                    // Trigger click with right button flag
                    canvas.dispatchEvent(new MouseEvent('click', {
                        button: 2,
                        clientX: e.clientX,
                        clientY: e.clientY,
                        shiftKey: e.shiftKey
                    }));
                });

                canvas.addEventListener("mousemove", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

                    const hovered = this.findPointAt(x, y);
                    if (hovered !== this.canvasWidget.hoveredPoint) {
                        this.canvasWidget.hoveredPoint = hovered;
                        this.redrawCanvas();
                    }
                });

                // Handle image input changes + video metadata
                this.onExecuted = (message) => {
                    console.log("[SAM3] onExecuted called with message:", message);

                    // Update video state from server
                    if (message.node_id?.[0])      this.videoState.nodeId      = message.node_id[0];
                    if (message.fps?.[0])           this.videoState.fps         = parseFloat(message.fps[0]);
                    if (message.total_frames?.[0]) {
                        this.videoState.totalFrames = parseInt(message.total_frames[0]);
                        this._videoControls.slider.max = this.videoState.totalFrames - 1;
                    }
                    if (message.frame_idx?.[0]) {
                        this._videoControls.updateFrameDisplay(parseInt(message.frame_idx[0]));
                    }

                    // Show/hide mask toggle based on whether a mask video is configured
                    if (message.has_mask_video?.[0] !== undefined) {
                        const hasMask = message.has_mask_video[0] === "1";
                        if (this._maskToggleBtn) {
                            this._maskToggleBtn.style.display = hasMask ? "" : "none";
                        }
                        if (!hasMask) {
                            this.canvasWidget.showMask = false;
                            this.canvasWidget.maskImage = null;
                        }
                    }

                    if (message.bg_image && message.bg_image[0]) {
                        const img = new Image();
                        img.onload = () => {
                            console.log(`[SAM3] Image loaded: ${img.width}x${img.height}`);
                            this.canvasWidget.image = img;
                            canvas.width  = img.width;
                            canvas.height = img.height;
                            this.canvasWidget.aspectRatio = img.width / img.height;
                            setCanvasHeight(Math.max(50, (this.size[0] - 20) / this.canvasWidget.aspectRatio));
                            fitHeight(this);
                            this.redrawCanvas();
                        };
                        img.src = "data:image/jpeg;base64," + message.bg_image[0];
                    }
                };



                // Draw initial placeholder
                console.log("[SAM3] Drawing initial placeholder");
                this.redrawCanvas();

                // Set initial node size (aspect ratio not known yet; uses 300px default)
                const nodeWidth = Math.max(400, this.size[0] || 400);
                this.size[0] = nodeWidth;
                setCanvasHeight(300);
                fitHeight(this);

                console.log("[SAM3] Node size set to:", this.size);
                console.log("[SAM3] onNodeCreated complete");
                return result;
            };

            // Helper: Clear canvas (called by downstream SAM3TwoMouseTracking)
            nodeType.prototype.clearCanvas = function() {
                this.canvasWidget.positivePoints = [];
                this.canvasWidget.negativePoints = [];
                this.updatePoints();
                this.redrawCanvas();
            };

            // Helper: Find point at coordinates
            nodeType.prototype.findPointAt = function(x, y) {
                const threshold = 10;

                for (const point of this.canvasWidget.positivePoints) {
                    if (Math.abs(point.x - x) < threshold && Math.abs(point.y - y) < threshold) {
                        return {type: 'positive', point};
                    }
                }

                for (const point of this.canvasWidget.negativePoints) {
                    if (Math.abs(point.x - x) < threshold && Math.abs(point.y - y) < threshold) {
                        return {type: 'negative', point};
                    }
                }

                return null;
            };

            // Helper: Update widget values
            nodeType.prototype.updatePoints = function() {
                // Use stored hidden widget references
                const coordsWidget = this._hiddenWidgets?.coordinates || this.widgets.find(w => w.name === "coordinates");
                const negCoordsWidget = this._hiddenWidgets?.neg_coordinates || this.widgets.find(w => w.name === "neg_coordinates");
                const storeWidget = this._hiddenWidgets?.points_store || this.widgets.find(w => w.name === "points_store");

                if (coordsWidget) {
                    coordsWidget.value = JSON.stringify(this.canvasWidget.positivePoints);
                }
                if (negCoordsWidget) {
                    negCoordsWidget.value = JSON.stringify(this.canvasWidget.negativePoints);
                }
                if (storeWidget) {
                    storeWidget.value = JSON.stringify({
                        positive: this.canvasWidget.positivePoints,
                        negative: this.canvasWidget.negativePoints
                    });
                }

                // Update points counter display
                const posCount = this.canvasWidget.positivePoints.length;
                const negCount = this.canvasWidget.negativePoints.length;
                this.canvasWidget.pointsCounter.textContent = `Points: ${posCount} pos, ${negCount} neg`;
            };

            // Helper: Redraw canvas
            nodeType.prototype.redrawCanvas = function() {
                const {canvas, ctx, image, maskImage, showMask, positivePoints, negativePoints, hoveredPoint} = this.canvasWidget;

                // Clear
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw image if available
                if (image) {
                    if (showMask && maskImage) {
                        // Show annotated (colorized) frame as background
                        ctx.drawImage(maskImage, 0, 0, canvas.width, canvas.height);
                    } else {
                        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                    }
                } else {
                    // Placeholder
                    ctx.fillStyle = "#333";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#666";
                    ctx.font = "16px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Click to add points", canvas.width / 2, canvas.height / 2);
                    ctx.fillText("Left-click: Positive (green)", canvas.width / 2, canvas.height / 2 + 25);
                    ctx.fillText("Shift/Right-click: Negative (red)", canvas.width / 2, canvas.height / 2 + 50);
                }

                // Draw canvas dimensions overlay (helpful for debugging)
                if (image) {
                    ctx.fillStyle = "rgba(0,0,0,0.7)";
                    ctx.fillRect(5, canvas.height - 25, 150, 20);
                    ctx.fillStyle = "#0f0";
                    ctx.font = "12px monospace";
                    ctx.textAlign = "left";
                    ctx.fillText(`Image: ${canvas.width}x${canvas.height}`, 10, canvas.height - 10);
                }

                // Draw positive points (green)
                ctx.strokeStyle = "#0f0";
                ctx.fillStyle = "#0f0";
                for (const point of positivePoints) {
                    const isHovered = hoveredPoint?.point === point;
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, isHovered ? 8 : 6, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }

                // Draw negative points (red)
                ctx.strokeStyle = "#f00";
                ctx.fillStyle = "#f00";
                for (const point of negativePoints) {
                    const isHovered = hoveredPoint?.point === point;
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, isHovered ? 8 : 6, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            };
        }
    }
});
