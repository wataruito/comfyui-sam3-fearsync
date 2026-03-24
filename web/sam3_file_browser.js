/**
 * Shared file-browser modal for SAM3 nodes.
 *
 * Usage:
 *   import { openFileBrowser } from "./sam3_file_browser.js";
 *   openFileBrowser({ node, widgetName, browseEndpoint, fileTypeLabel, onPicked });
 */

async function _fetchDir(endpoint, path) {
    const url = `${endpoint}${path ? "?path=" + encodeURIComponent(path) : ""}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
}

async function _fetchPaths() {
    try {
        const r = await fetch("/sam3/get_paths");
        return r.ok ? r.json() : {};
    } catch { return {}; }
}

function _fmtSize(b) {
    if (b > 1e9) return (b / 1e9).toFixed(1) + " GB";
    if (b > 1e6) return (b / 1e6).toFixed(1) + " MB";
    if (b > 1e3) return (b / 1e3).toFixed(1) + " KB";
    return b + " B";
}

/**
 * Open a file-browser modal dialog.
 *
 * @param {object} opts
 * @param {object}   opts.node            LiteGraph node
 * @param {string}   opts.widgetName      Widget whose .value is set on confirm
 * @param {string}   opts.browseEndpoint  API endpoint (e.g. "/sam3/browse_files")
 * @param {string}  [opts.fileTypeLabel]  Type column label, default "File"
 * @param {Function}[opts.onPicked]       Called with selectedPath after confirm
 */
export function openFileBrowser({ node, widgetName, browseEndpoint,
                                  fileTypeLabel = "File", initialPath = null, onPicked }) {
    let selectedPath = null;

    // ── overlay ──────────────────────────────────────────────────────────
    const overlay = document.createElement("div");
    overlay.style.cssText =
        "position:fixed;inset:0;background:rgba(0,0,0,0.75);" +
        "display:flex;align-items:center;justify-content:center;z-index:99999;";

    // ── dialog ───────────────────────────────────────────────────────────
    const dlg = document.createElement("div");
    dlg.style.cssText =
        "background:#2b2b2b;border:1px solid #555;border-radius:8px;" +
        "width:780px;max-width:92vw;height:560px;max-height:88vh;" +
        "display:flex;flex-direction:column;overflow:hidden;" +
        "box-shadow:0 12px 40px rgba(0,0,0,0.9);";

    // ── header ───────────────────────────────────────────────────────────
    const hdr = document.createElement("div");
    hdr.style.cssText =
        "display:flex;align-items:center;padding:10px 14px;gap:8px;" +
        "background:#1e1e1e;border-bottom:1px solid #444;flex-shrink:0;";

    const cancelB = document.createElement("button");
    cancelB.textContent = "Cancel";
    cancelB.style.cssText =
        "padding:3px 12px;background:#3a3a3a;color:#ccc;" +
        "border:1px solid #555;border-radius:5px;cursor:pointer;font-size:12px;";
    cancelB.onmouseover = () => cancelB.style.background = "#484848";
    cancelB.onmouseout  = () => cancelB.style.background = "#3a3a3a";

    const titleEl = document.createElement("span");
    titleEl.textContent = "Open File";
    titleEl.style.cssText =
        "flex:1;text-align:center;font-size:14px;font-weight:600;" +
        "color:#fff;font-family:sans-serif;";

    const selectB = document.createElement("button");
    selectB.textContent = "Select";
    selectB.style.cssText =
        "padding:3px 18px;background:#c0392b;color:#fff;" +
        "border:none;border-radius:5px;cursor:pointer;font-size:12px;" +
        "font-weight:600;opacity:0.35;pointer-events:none;";

    hdr.append(cancelB, titleEl, selectB);

    // ── body: sidebar + main ─────────────────────────────────────────────
    const body = document.createElement("div");
    body.style.cssText = "display:flex;flex:1;overflow:hidden;";

    const sidebar = document.createElement("div");
    sidebar.style.cssText =
        "width:130px;flex-shrink:0;background:#202020;" +
        "border-right:1px solid #353535;overflow-y:auto;padding:10px 0;";

    const main = document.createElement("div");
    main.style.cssText = "flex:1;display:flex;flex-direction:column;overflow:hidden;";

    // breadcrumb
    const crumb = document.createElement("div");
    crumb.style.cssText =
        "padding:5px 12px;background:#1a1a1a;border-bottom:1px solid #303030;" +
        "font-size:11px;font-family:monospace;color:#888;flex-shrink:0;" +
        "white-space:nowrap;overflow-x:auto;display:flex;align-items:center;gap:2px;";

    // column headers
    const colHdr = document.createElement("div");
    colHdr.style.cssText =
        "display:flex;align-items:center;padding:4px 12px;" +
        "background:#1a1a1a;border-bottom:1px solid #303030;flex-shrink:0;" +
        "font-size:10px;color:#666;font-family:sans-serif;user-select:none;";
    const mkColHdr = (t, w) => {
        const s = document.createElement("span");
        s.textContent = t;
        s.style.cssText = w ? `width:${w};text-align:right;flex-shrink:0;` : "flex:1;";
        return s;
    };
    const typeH = mkColHdr("Type", "70px");
    typeH.style.textAlign = "center";
    colHdr.append(mkColHdr("Name"), mkColHdr("Size", "70px"), typeH, mkColHdr("Modified", "130px"));

    // file list
    const fileList = document.createElement("div");
    fileList.style.cssText = "flex:1;overflow-y:auto;";

    // footer / status
    const footer = document.createElement("div");
    footer.style.cssText =
        "padding:6px 12px;background:#1a1a1a;border-top:1px solid #303030;" +
        "font-size:10px;font-family:monospace;color:#666;flex-shrink:0;" +
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    footer.textContent = "Select a file";

    main.append(crumb, colHdr, fileList, footer);
    body.append(sidebar, main);
    dlg.append(hdr, body);
    overlay.appendChild(dlg);
    document.body.appendChild(overlay);

    // ── navigate ─────────────────────────────────────────────────────────
    async function navigate(path) {
        fileList.innerHTML =
            '<div style="padding:24px;text-align:center;color:#555;' +
            'font-family:sans-serif;font-size:12px;">Loading…</div>';
        crumb.innerHTML = "";
        try {
            const data = await _fetchDir(browseEndpoint, path);
            buildCrumb(data.current);
            fileList.innerHTML = "";
            if (data.parent) {
                fileList.appendChild(makeRow("⬆  ..", null, "parent", data.parent));
            }
            data.entries.forEach(e => {
                fileList.appendChild(makeRow(
                    e.type === "dir" ? `📁  ${e.name}` : `📄  ${e.name}`,
                    e, e.type, e.path,
                ));
            });
            if (data.entries.length === 0 && !data.parent) {
                const msg = document.createElement("div");
                msg.style.cssText =
                    "padding:24px;text-align:center;color:#444;" +
                    "font-size:12px;font-family:sans-serif;";
                msg.textContent = "No files found here.";
                fileList.appendChild(msg);
            }
        } catch (err) {
            fileList.innerHTML =
                `<div style="padding:16px;color:#f66;font-size:12px;` +
                `font-family:monospace;">${err.message}</div>`;
        }
    }

    function buildCrumb(fullPath) {
        crumb.innerHTML = "";
        const parts = fullPath.split("/").filter(Boolean);
        let acc = "";
        parts.forEach((part, i) => {
            if (i > 0) {
                const sep = document.createElement("span");
                sep.textContent = " › ";
                sep.style.color = "#3a3a3a";
                crumb.appendChild(sep);
            }
            acc += "/" + part;
            const sp = document.createElement("span");
            sp.textContent = part;
            const captured = acc;
            if (i < parts.length - 1) {
                sp.style.cssText = "cursor:pointer;color:#5a8fc0;";
                sp.onmouseover = () => sp.style.color = "#7ab0e0";
                sp.onmouseout  = () => sp.style.color = "#5a8fc0";
                sp.onclick = () => navigate(captured);
            } else {
                sp.style.color = "#ccc";
            }
            crumb.appendChild(sp);
        });
    }

    function makeRow(text, entry, type, path) {
        const row = document.createElement("div");
        row.style.cssText =
            "display:flex;align-items:center;padding:5px 12px;" +
            "border-bottom:1px solid #1c1c1c;cursor:pointer;" +
            "font-family:monospace;font-size:12px;color:#bbb;";

        const nameSp = document.createElement("span");
        nameSp.textContent = text;
        nameSp.style.cssText =
            "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        row.appendChild(nameSp);

        if (type === "file" && entry) {
            nameSp.style.color = "#ddd";

            const sizeSp = document.createElement("span");
            sizeSp.textContent = _fmtSize(entry.size);
            sizeSp.style.cssText =
                "width:70px;text-align:right;color:#777;font-size:11px;flex-shrink:0;";

            const typeSp = document.createElement("span");
            typeSp.textContent = fileTypeLabel;
            typeSp.style.cssText =
                "width:70px;text-align:center;color:#777;font-size:11px;flex-shrink:0;";

            const dateSp = document.createElement("span");
            dateSp.textContent = new Date(entry.mtime * 1000).toLocaleDateString();
            dateSp.style.cssText =
                "width:130px;text-align:right;color:#777;font-size:11px;flex-shrink:0;";

            row.append(sizeSp, typeSp, dateSp);

            row.onclick = () => {
                fileList.querySelectorAll("[data-sel]").forEach(r => {
                    r.style.background = "";
                    delete r.dataset.sel;
                });
                row.style.background = "#1b3a5e";
                row.dataset.sel = "1";
                selectedPath = path;
                selectB.style.opacity = "1";
                selectB.style.pointerEvents = "auto";
                footer.textContent = path;
                footer.style.color = "#aaa";
            };
            row.ondblclick = () => { row.click(); confirmSelect(); };
        } else {
            // directory
            nameSp.style.color = "#5a8fc0";
            row.onclick    = () => navigate(path);
            row.ondblclick = () => navigate(path);
        }

        row.onmouseover = () => { if (!row.dataset.sel) row.style.background = "#2a2a2a"; };
        row.onmouseout  = () => { if (!row.dataset.sel) row.style.background = ""; };
        return row;
    }

    // ── sidebar bookmarks ─────────────────────────────────────────────────
    async function buildSidebar() {
        const paths = await _fetchPaths();

        const sep = document.createElement("div");
        sep.style.cssText =
            "padding:4px 12px;font-size:9px;color:#444;font-family:sans-serif;" +
            "text-transform:uppercase;letter-spacing:1px;margin-top:2px;";
        sep.textContent = "Bookmarks";
        sidebar.appendChild(sep);

        const bookmarks = [
            { label: "Recent", icon: "🕐", path: null },
            { label: "Home",   icon: "🏠", path: paths.home },
            { label: "Output", icon: "📤", path: paths.output },
        ];
        bookmarks.forEach(bk => {
            if (!bk.path && bk.label !== "Recent") return;
            const item = document.createElement("div");
            item.style.cssText =
                "display:flex;align-items:center;gap:7px;padding:7px 14px;" +
                "cursor:pointer;font-size:12px;color:#999;font-family:sans-serif;";
            const icon = document.createElement("span");
            icon.textContent = bk.icon;
            icon.style.fontSize = "14px";
            const lbl = document.createElement("span");
            lbl.textContent = bk.label;
            item.append(icon, lbl);
            item.onmouseover = () => item.style.background = "#2a2a2a";
            item.onmouseout  = () => item.style.background = "";
            item.onclick = () => navigate(bk.path);
            sidebar.appendChild(item);
        });
    }

    // ── confirm / close ───────────────────────────────────────────────────
    function confirmSelect() {
        if (!selectedPath) return;
        const pw = node.widgets?.find(w => w.name === widgetName);
        if (pw) {
            pw.value = selectedPath;
            const el = pw.element || pw.inputEl;
            if (el) el.value = selectedPath;
        }
        onPicked?.(selectedPath);
        close();
    }

    function close() { overlay.remove(); }

    cancelB.onclick = close;
    selectB.onclick = confirmSelect;
    overlay.onclick = e => { if (e.target === overlay) close(); };
    document.addEventListener("keydown", function esc(e) {
        if (e.key === "Escape") {
            close();
            document.removeEventListener("keydown", esc);
        }
    });

    buildSidebar();
    navigate(initialPath);  // start at initialPath, or output directory if null
}

// ── Directory picker modal ────────────────────────────────────────────────
//
// Opens a "Choose Folder" dialog. Only directories are shown.
// Navigating into a directory and clicking "Select This Folder" returns
// the path relative to the ComfyUI output directory.
//
// @param {object} opts
// @param {object}   opts.node        LiteGraph node
// @param {string}   opts.widgetName  Widget whose .value is set on confirm
// @param {Function}[opts.onPicked]   Called with relative path after confirm

export function openDirBrowser({ node, widgetName, onPicked }) {
    let currentPath = null;
    let outputDir   = null;

    // ── overlay ──────────────────────────────────────────────────────────
    const overlay = document.createElement("div");
    overlay.style.cssText =
        "position:fixed;inset:0;background:rgba(0,0,0,0.75);" +
        "display:flex;align-items:center;justify-content:center;z-index:99999;";

    // ── dialog ───────────────────────────────────────────────────────────
    const dlg = document.createElement("div");
    dlg.style.cssText =
        "background:#2b2b2b;border:1px solid #555;border-radius:8px;" +
        "width:620px;max-width:90vw;height:480px;max-height:85vh;" +
        "display:flex;flex-direction:column;overflow:hidden;" +
        "box-shadow:0 12px 40px rgba(0,0,0,0.9);";

    // ── header ───────────────────────────────────────────────────────────
    const hdr = document.createElement("div");
    hdr.style.cssText =
        "display:flex;align-items:center;padding:10px 14px;gap:8px;" +
        "background:#1e1e1e;border-bottom:1px solid #444;flex-shrink:0;";

    const cancelB = document.createElement("button");
    cancelB.textContent = "Cancel";
    cancelB.style.cssText =
        "padding:3px 12px;background:#3a3a3a;color:#ccc;" +
        "border:1px solid #555;border-radius:5px;cursor:pointer;font-size:12px;";
    cancelB.onmouseover = () => cancelB.style.background = "#484848";
    cancelB.onmouseout  = () => cancelB.style.background = "#3a3a3a";

    const titleEl = document.createElement("span");
    titleEl.textContent = "Choose Folder";
    titleEl.style.cssText =
        "flex:1;text-align:center;font-size:14px;font-weight:600;" +
        "color:#fff;font-family:sans-serif;";

    const selectB = document.createElement("button");
    selectB.textContent = "Select This Folder";
    selectB.style.cssText =
        "padding:3px 14px;background:#c0392b;color:#fff;" +
        "border:none;border-radius:5px;cursor:pointer;font-size:12px;" +
        "font-weight:600;white-space:nowrap;opacity:0.35;pointer-events:none;";

    hdr.append(cancelB, titleEl, selectB);

    // ── body: sidebar + main ─────────────────────────────────────────────
    const body = document.createElement("div");
    body.style.cssText = "display:flex;flex:1;overflow:hidden;";

    const sidebar = document.createElement("div");
    sidebar.style.cssText =
        "width:130px;flex-shrink:0;background:#202020;" +
        "border-right:1px solid #353535;overflow-y:auto;padding:10px 0;";

    const main = document.createElement("div");
    main.style.cssText = "flex:1;display:flex;flex-direction:column;overflow:hidden;";

    // breadcrumb
    const crumb = document.createElement("div");
    crumb.style.cssText =
        "padding:5px 12px;background:#1a1a1a;border-bottom:1px solid #303030;" +
        "font-size:11px;font-family:monospace;color:#888;flex-shrink:0;" +
        "white-space:nowrap;overflow-x:auto;display:flex;align-items:center;gap:2px;";

    // directory list
    const dirList = document.createElement("div");
    dirList.style.cssText = "flex:1;overflow-y:auto;";

    // footer: shows selected relative path
    const footer = document.createElement("div");
    footer.style.cssText =
        "padding:6px 12px;background:#1a1a1a;border-top:1px solid #303030;" +
        "font-size:10px;font-family:monospace;color:#666;flex-shrink:0;" +
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    footer.textContent = "Navigate to a folder, then click \"Select This Folder\"";

    main.append(crumb, dirList, footer);
    body.append(sidebar, main);
    dlg.append(hdr, body);
    overlay.appendChild(dlg);
    document.body.appendChild(overlay);

    // ── navigate ─────────────────────────────────────────────────────────
    async function navigate(path) {
        dirList.innerHTML =
            '<div style="padding:24px;text-align:center;color:#555;' +
            'font-family:sans-serif;font-size:12px;">Loading…</div>';
        crumb.innerHTML = "";
        try {
            const url = `/sam3/browse_dirs${path ? "?path=" + encodeURIComponent(path) : ""}`;
            const r = await fetch(url);
            if (!r.ok) throw new Error(await r.text());
            const data = await r.json();

            currentPath = data.current;
            if (!outputDir && data.output_dir) outputDir = data.output_dir;

            // Enable Select button now that we have a valid path
            selectB.style.opacity = "1";
            selectB.style.pointerEvents = "auto";

            // Update footer with relative path
            const rel = outputDir ? relPath(data.current, outputDir) : data.current;
            footer.textContent = rel ? `Output prefix: "${rel}"` : "(output root — type a name)";
            footer.style.color = "#aaa";

            buildCrumb(data.current);

            dirList.innerHTML = "";
            if (data.parent) {
                dirList.appendChild(makeDirRow("⬆  ..", data.parent, true));
            }
            data.entries.forEach(e => {
                dirList.appendChild(makeDirRow(`📁  ${e.name}`, e.path, false));
            });
            if (data.entries.length === 0 && !data.parent) {
                const msg = document.createElement("div");
                msg.style.cssText =
                    "padding:24px;text-align:center;color:#444;" +
                    "font-size:12px;font-family:sans-serif;";
                msg.textContent = "No subdirectories here.";
                dirList.appendChild(msg);
            }
        } catch (err) {
            dirList.innerHTML =
                `<div style="padding:16px;color:#f66;font-size:12px;` +
                `font-family:monospace;">${err.message}</div>`;
        }
    }

    function relPath(absPath, base) {
        // Return path relative to base, or "" if equal
        const norm = (p) => p.replace(/\/+$/, "");
        absPath = norm(absPath);
        base    = norm(base);
        if (absPath === base) return "";
        if (absPath.startsWith(base + "/")) return absPath.slice(base.length + 1);
        return absPath;  // outside output dir — return as-is
    }

    function buildCrumb(fullPath) {
        crumb.innerHTML = "";
        const parts = fullPath.split("/").filter(Boolean);
        let acc = "";
        parts.forEach((part, i) => {
            if (i > 0) {
                const sep = document.createElement("span");
                sep.textContent = " › ";
                sep.style.color = "#3a3a3a";
                crumb.appendChild(sep);
            }
            acc += "/" + part;
            const sp = document.createElement("span");
            sp.textContent = part;
            const captured = acc;
            if (i < parts.length - 1) {
                sp.style.cssText = "cursor:pointer;color:#5a8fc0;";
                sp.onmouseover = () => sp.style.color = "#7ab0e0";
                sp.onmouseout  = () => sp.style.color = "#5a8fc0";
                sp.onclick = () => navigate(captured);
            } else {
                sp.style.color = "#ccc";
            }
            crumb.appendChild(sp);
        });
    }

    function makeDirRow(text, path, isParent) {
        const row = document.createElement("div");
        row.style.cssText =
            "display:flex;align-items:center;padding:7px 14px;" +
            "border-bottom:1px solid #1c1c1c;cursor:pointer;" +
            "font-family:monospace;font-size:12px;";

        const nameSp = document.createElement("span");
        nameSp.textContent = text;
        nameSp.style.color = isParent ? "#888" : "#5a8fc0";
        row.appendChild(nameSp);

        row.onclick    = () => navigate(path);
        row.ondblclick = () => navigate(path);
        row.onmouseover = () => row.style.background = "#2a2a2a";
        row.onmouseout  = () => row.style.background = "";
        return row;
    }

    // ── sidebar bookmarks ─────────────────────────────────────────────────
    async function buildSidebar() {
        const paths = await _fetchPaths();
        const sep = document.createElement("div");
        sep.style.cssText =
            "padding:4px 12px;font-size:9px;color:#444;font-family:sans-serif;" +
            "text-transform:uppercase;letter-spacing:1px;margin-top:2px;";
        sep.textContent = "Bookmarks";
        sidebar.appendChild(sep);

        const bookmarks = [
            { label: "Output", icon: "📤", path: paths.output },
        ];
        bookmarks.forEach(bk => {
            if (!bk.path) return;
            const item = document.createElement("div");
            item.style.cssText =
                "display:flex;align-items:center;gap:7px;padding:7px 14px;" +
                "cursor:pointer;font-size:12px;color:#999;font-family:sans-serif;";
            const icon = document.createElement("span");
            icon.textContent = bk.icon;
            icon.style.fontSize = "14px";
            const lbl = document.createElement("span");
            lbl.textContent = bk.label;
            item.append(icon, lbl);
            item.onmouseover = () => item.style.background = "#2a2a2a";
            item.onmouseout  = () => item.style.background = "";
            item.onclick = () => navigate(bk.path);
            sidebar.appendChild(item);
        });
    }

    // ── confirm / close ───────────────────────────────────────────────────
    function confirmSelect() {
        if (!currentPath || !outputDir) return;
        const rel = relPath(currentPath, outputDir);
        // Guard: reject paths outside the output directory
        if (rel.startsWith("/") || rel.startsWith("..")) {
            footer.textContent = "⚠ Please select a folder inside the output directory.";
            footer.style.color = "#f88";
            return;
        }
        const pw = node.widgets?.find(w => w.name === widgetName);
        if (pw) {
            pw.value = rel;
            const el = pw.element || pw.inputEl;
            if (el) el.value = rel;
        }
        onPicked?.(rel);
        close();
    }

    function close() { overlay.remove(); }

    cancelB.onclick = close;
    selectB.onclick = confirmSelect;
    overlay.onclick = e => { if (e.target === overlay) close(); };
    document.addEventListener("keydown", function esc(e) {
        if (e.key === "Escape") {
            close();
            document.removeEventListener("keydown", esc);
        }
    });

    buildSidebar();
    navigate(null);  // start at output directory
}
