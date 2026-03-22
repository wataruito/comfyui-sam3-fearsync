# SAM3AnimalPointCollector Resize Fix Log

Date: 2026-03-21

---

## Problem Summary

`SAM3AnimalPointCollector` node had two resize-related bugs:

1. **JS not loading at all** — canvas, slider, and file selector were not shown; hidden widgets (`points_store`, `coordinates`, `neg_coordinates`) were visible
2. **Auto-expanding loop** — after loading a video frame, the node kept growing in height

---

## Step 1: Extension name conflict (root cause of missing UI)

**File:** `web/sam3_animal_point_widget.js`

**Problem:**
Both `sam3_animal_point_widget.js` and `sam3_points_widget.js` registered an extension named `"Comfy.SAM3.SimplePointCollector"`. ComfyUI only registers one extension per name, so the Animal variant's JS was silently ignored. All DOM widgets (canvas, slider, file selector) were never created.

**Fix:**
```js
// Before
app.registerExtension({ name: "Comfy.SAM3.SimplePointCollector", ...

// After
app.registerExtension({ name: "Comfy.SAM3.AnimalPointCollector", ...
```

---

## Step 2: Remove `setSize()` from `onResize`

**File:** `web/sam3_animal_point_widget.js`

**Problem:**
The original `onResize` called `this.setSize()`, which caused LiteGraph to fire `onResize` again — a potential oscillation / infinite loop.

**Fix:**
Removed `setSize()` from `onResize`. Canvas height is derived from `size[1]` directly:

```js
// Before (inside onResize)
this.setSize([nodeWidth, this.computeSize()[1]]);
this._isResizing = false;  // synchronous reset

// After
// setSize() removed entirely from onResize
setTimeout(() => { this._isResizing = false; }, 50);  // delayed reset
```

Also moved `_isResizing` guard to after `originalOnResize` call (matching `SAM3PointCollector` pattern).

---

## Step 3: `setTimeout` for `_isResizing` reset in `fetchFrame` / `onExecuted`

**File:** `web/sam3_animal_point_widget.js`

**Problem:**
In `fetchFrame` (slider scrub) and `onExecuted` (after queue), `_isResizing = false` was reset synchronously after `setSize()`. LiteGraph can call `onResize` asynchronously (next animation frame) after `setSize()`. By that time `_isResizing` was already `false`, so the guard was bypassed, causing another round of updates.

**Fix:**
```js
// Before
this.setSize([nodeWidth, this.computeSize()[1]]);
this._isResizing = false;  // ← fired before LiteGraph's async onResize

// After
this.setSize([nodeWidth, newH + this._canvasOverhead]);
setTimeout(() => { this._isResizing = false; }, 50);  // ← covers async call
```

---

## Step 4: Replace `computeSize()[1]` with fixed formula in `setSize` calls

**File:** `web/sam3_animal_point_widget.js`

**Problem:**
Using `this.computeSize()[1]` inside `setSize()` is self-referential: the result includes LiteGraph-internal gaps that are not accounted for in the `onResize` subtraction formula (`size[1] - overhead`). This mismatch caused `onResize` to compute a slightly larger `widgetHeight` each time → expansion loop.

**Fix:**
All `setSize` calls now use a fixed formula: `newH + this._canvasOverhead`, where `_canvasOverhead = 136` (see Step 5).

```js
// Before
this.setSize([nodeWidth, this.computeSize()[1]]);

// After
this.setSize([nodeWidth, newH + this._canvasOverhead]);
```

---

## Step 5: Fixed overhead constant; `onResize` only updates container visually

**File:** `web/sam3_animal_point_widget.js`

**Problem:**
Attempting to measure `_canvasOverhead` dynamically at `onNodeCreated` time (`this.computeSize()[1] - widgetHeight`) was inaccurate because DOM widgets had not yet been rendered by the browser. This produced a wrong (too small) overhead, causing the `onResize` formula `size[1] - overhead` to compute a `visH` that was larger than intended — pushing slider and file selector out of the visible area.

Also, updating `widgetHeight` inside `onResize` created a `computeSize` feedback loop:
`onResize` → `widgetHeight` increases → `computeSize()[1]` increases → LiteGraph calls `onResize` again → loop.

**Fix:**
1. Use a **fixed constant** `_canvasOverhead = 136`:
   - `80` = title + padding (same as `SAM3PointCollector`)
   - `30` = video controls DOM widget
   - `26` = file selector DOM widget
2. `onResize` **does not update `widgetHeight`** — it only adjusts `container.style.height` for visual scaling:

```js
this.onResize = function(size) {
    if (originalOnResize) originalOnResize.apply(this, arguments);
    if (this._isResizing) return;
    const visH = Math.max(50, size[1] - this._canvasOverhead);
    container.style.height = visH + "px";
    this.redrawCanvas();
};
```

**Remaining limitation:**
Because `widgetHeight` is not updated in `onResize`, LiteGraph enforces a minimum node height equal to `widgetHeight + overhead` (the last programmatically set height). The user **cannot shrink the node below the image's natural aspect-ratio height**. This is a LiteGraph minimum-size constraint, not a bug introduced here.

---

## Summary of Current Behavior (after all fixes)

| Action | Behavior |
|--------|----------|
| Node creation | Canvas (300px) + controls + file selector correctly laid out |
| Load video frame (slider/queue) | Node resizes once to match image aspect ratio; no loop |
| Manual expand | Node grows; canvas follows |
| Manual shrink | Limited to `widgetHeight + 136` (LiteGraph minimum) |
| Auto-expand loop | Fixed — `onResize` no longer updates `widgetHeight` |
