const E = {
  ERROR: "error",
  READY: "ready",
  LOADING: "loading"
}, ct = {
  aggregate: "auto",
  join: null,
  compat: null,
  data_vars: null,
  coords: null,
  dim: "",
  group_by: "",
  reload: !1,
  access_pattern: "map",
  chunk_size: 16,
  map_primary_chunksize: 1,
  timeout: 120
}, pt = "https://gridlook.pages.dev/";
function H(r) {
  return `${pt}#${r}`;
}
function ut(r) {
  var e;
  const t = navigator;
  return (e = t.clipboard) != null && e.writeText ? t.clipboard.writeText(r) : new Promise((i, a) => {
    try {
      const o = document.createElement("textarea");
      o.value = r, o.style.position = "fixed", o.style.opacity = "0", document.body.appendChild(o), o.focus(), o.select(), document.execCommand("copy"), document.body.removeChild(o), i();
    } catch (o) {
      a(o instanceof Error ? o : new Error(String(o)));
    }
  });
}
function v(r, t = !1) {
  return `<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false" style="vertical-align:-0.14em;flex-shrink:0${t ? ";margin-right:.5em" : ""}">${r}</svg>`;
}
const m = {
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11.5v5"/><circle cx="12" cy="7.8" r="0.6" fill="currentColor" stroke="none"/>',
  layers: '<path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="m3 13 9 5 9-5"/>',
  load: '<path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 4v4h-4"/>',
  chevronDown: '<path d="m6 9 6 6 6-6"/>',
  ban: '<circle cx="12" cy="12" r="9"/><path d="m5.6 5.6 12.8 12.8"/>',
  link: '<path d="M9.5 13.5a4 4 0 0 0 5.7 0l2.8-2.8a4 4 0 1 0-5.7-5.7l-1 1"/><path d="M14.5 10.5a4 4 0 0 0-5.7 0l-2.8 2.8a4 4 0 1 0 5.7 5.7l1-1"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  database: '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v13c0 1.66 3.58 3 8 3s8-1.34 8-3v-13"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/>',
  cube: '<path d="M12 3 3 7.5v9L12 21l9-4.5v-9L12 3Z"/><path d="m3 7.5 9 4.5 9-4.5"/><path d="M12 12v9"/>',
  external: '<path d="M14 4h6v6"/><path d="M20 4 11 13"/><path d="M19 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4"/>',
  refresh: '<path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 4v4h-4"/>',
  compress: '<path d="M9 5v4H5"/><path d="m4 4 5 5"/><path d="M15 5v4h4"/><path d="m20 4-5 5"/><path d="M9 19v-4H5"/><path d="m4 20 5-5"/><path d="M15 19v-4h4"/><path d="m20 20-5-5"/>',
  alert: '<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5"/><circle cx="12" cy="16.3" r="0.6" fill="currentColor" stroke="none"/>'
}, ht = `
data-inspector{
  --_di-bg:var(--di-bg,#fff);
  --_di-fg:var(--di-fg,#1f2937);
  --_di-muted:var(--di-muted,#6b7280);
  --_di-border:var(--di-border,#e5e7eb);
  --_di-surface:var(--di-surface,#f3f4f6);
  --_di-accent:var(--di-accent,#3b82f6);
}
@media (prefers-color-scheme:dark){
  data-inspector{
    --_di-bg:var(--di-bg,#1e293b);
    --_di-fg:var(--di-fg,#e5e7eb);
    --_di-muted:var(--di-muted,#94a3b8);
    --_di-border:var(--di-border,#475569);
    --_di-surface:var(--di-surface,#334155);
    --_di-accent:var(--di-accent,#3b82f6);
  }
}
.di-backdrop{position:fixed;inset:0;z-index:1050;display:flex;align-items:center;justify-content:center;padding:12px;background:rgba(15,23,42,.55);}
.di-modal{display:flex;flex-direction:column;width:min(1100px,96vw);max-height:95vh;overflow:hidden;background:var(--_di-bg);color:var(--_di-fg);border-radius:12px;box-shadow:0 20px 50px rgba(0,0,0,.25);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:14px;line-height:1.5;}
.di-header{flex-shrink:0;border-bottom:1px solid var(--_di-border);padding:16px 16px 12px;}
.di-header-row{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;}
.di-header-main{flex:1;min-width:0;}
.di-title{margin:0 0 12px;font-size:clamp(16px,4vw,20px);font-weight:600;display:flex;align-items:center;gap:8px;}
.di-title-ico{color:var(--_di-accent);display:inline-flex;align-items:center;font-size:16px;}
.di-close{flex-shrink:0;width:32px;height:32px;font-size:28px;line-height:1;background:transparent;border:none;border-radius:6px;color:var(--_di-muted);cursor:pointer;}
.di-close:hover{background:var(--_di-surface);color:var(--_di-fg);}
.di-muted{color:var(--_di-muted);}
.di-center{text-align:center;padding:24px 12px;}
.di-file-list{margin-bottom:12px;padding:10px;background:var(--_di-surface);border-radius:6px;max-height:120px;overflow-y:auto;}
.di-file-list-label{font-size:12px;color:var(--_di-muted);margin-bottom:6px;font-weight:500;}
.di-file-list ul{font-size:11px;margin:0;padding-left:20px;}
.di-file-list li{color:var(--_di-fg);word-break:break-all;}
.di-pathbar{margin-bottom:8px;}
.di-pathbar-label{display:block;font-size:12px;color:var(--_di-muted);margin-bottom:4px;font-weight:500;}
.di-pathbar-row{display:flex;gap:6px;flex-wrap:wrap;}
.di-input{flex:1 1 200px;min-width:0;font-size:13px;padding:6px 10px;border:1px solid var(--_di-border);border-radius:6px;color:var(--_di-fg);background:var(--_di-bg);}
.di-input:focus{outline:2px solid var(--_di-accent);outline-offset:0;border-color:var(--_di-accent);}
.di-dropdown-wrap{position:relative;flex-shrink:0;display:inline-flex;}
.di-btn{display:inline-flex;align-items:center;justify-content:center;font-size:13px;font-weight:500;border-radius:6px;border:1px solid transparent;padding:6px 12px;cursor:pointer;white-space:nowrap;background:var(--_di-bg);color:var(--_di-fg);}
.di-btn:disabled{opacity:.5;cursor:not-allowed;}
.di-btn-primary{background:var(--_di-accent);border-color:var(--_di-accent);color:#fff;}
.di-btn-primary:hover:not(:disabled){filter:brightness(.93);}
.di-btn-secondary{background:var(--_di-surface);border-color:var(--_di-surface);color:var(--_di-fg);}
.di-btn-outline{background:var(--_di-bg);border-color:var(--_di-border);color:var(--_di-fg);}
.di-btn-split{padding:6px 9px;border-top-left-radius:0;border-bottom-left-radius:0;border-left-color:rgba(255,255,255,.4);}
.di-btn-group{display:inline-flex;}
.di-btn-group>.di-btn:first-child{border-top-right-radius:0;border-bottom-right-radius:0;}
.di-menu{position:absolute;top:100%;right:0;z-index:1060;min-width:210px;margin-top:2px;padding:6px 0;background:var(--_di-bg);border:1px solid var(--_di-border);border-radius:8px;box-shadow:0 10px 25px rgba(0,0,0,.12);list-style:none;}
.di-menu-item{display:flex;align-items:center;width:100%;padding:8px 14px;font-size:13px;background:none;border:none;color:var(--_di-fg);cursor:pointer;text-align:left;}
.di-menu-item:hover{background:var(--_di-surface);}
.di-zarr-row{display:flex;align-items:center;flex-wrap:wrap;gap:6px;padding:8px 10px;background:var(--_di-surface);border-radius:6px;font-size:11px;margin-bottom:8px;}
.di-zarr-inner{display:flex;align-items:center;gap:6px;flex:1 1 100%;min-width:0;}
.di-code{flex:1;min-width:0;background:var(--_di-bg);padding:4px 8px;border-radius:4px;font-size:10px;color:var(--_di-fg);border:1px solid var(--_di-border);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.di-tabs{display:flex;gap:2px;border-bottom:2px solid var(--_di-border);}
.di-tab{padding:12px 16px;font-size:14px;font-weight:500;background:transparent;border:none;border-bottom:3px solid transparent;color:var(--_di-muted);cursor:pointer;transition:all .15s ease;display:inline-flex;align-items:center;}
.di-tab:disabled{opacity:.5;cursor:not-allowed;}
.di-tab-active{font-weight:600;background:var(--_di-surface);border-bottom-color:var(--_di-accent);color:var(--_di-fg);}
.di-body{flex:1;overflow-y:auto;overflow-x:hidden;padding:16px 12px;max-height:calc(95vh - 200px);}
.di-error{border-radius:8px;background:#fee2e2;border:1px solid #fecaca;color:#991b1b;font-size:13px;padding:12px;margin-bottom:16px;}
.di-error-row{display:flex;align-items:flex-start;gap:10px;}
.di-error-ico{font-size:18px;margin-top:1px;flex-shrink:0;}
.di-error-body{flex:1;min-width:0;}
.di-error-title{display:block;margin-bottom:6px;}
.di-error-msg{word-wrap:break-word;overflow-wrap:anywhere;white-space:pre-wrap;}
.di-btn-danger{background:#dc2626;border-color:#dc2626;color:#fff;font-size:12px;padding:6px 12px;margin-top:8px;}
.di-metadata{display:flex;justify-content:flex-start;width:100%;overflow-x:auto;}
.di-metadata>*{width:100%;min-width:0;}
.di-metadata dd,.di-metadata .xr-attrs td,.di-metadata .xr-var-attrs td{overflow-wrap:anywhere;word-break:break-word;}
.di-gridlook-bar{display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:12px;background:var(--_di-surface);border-radius:8px;font-size:12px;margin-bottom:16px;border:1px solid var(--_di-border);}
.di-gridlook-inner{display:flex;align-items:center;gap:6px;flex:1 1 100%;min-width:0;}
.di-gridlook-ico{color:var(--_di-accent);flex-shrink:0;display:inline-flex;}
.di-gridlook-label{color:var(--_di-accent);font-weight:600;flex-shrink:0;}
.di-gridlook-code{flex:1;min-width:0;background:var(--_di-bg);padding:6px 10px;border-radius:4px;font-size:11px;color:var(--_di-fg);border:1px solid var(--_di-border);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.di-gridlook-btn{background:var(--_di-bg);border:1px solid var(--_di-accent);color:var(--_di-accent);font-size:12px;padding:6px 12px;flex-shrink:0;}
.di-gridlook-btn-primary{background:var(--_di-accent);border-color:var(--_di-accent);color:#fff;}
.di-gridlook-frame{width:100%;height:calc(95vh - 280px);min-height:500px;background:var(--_di-surface);border-radius:8px;overflow:hidden;border:1px solid var(--_di-border);}
.di-gridlook-frame iframe{width:100%;height:100%;border:none;display:block;}
.di-agg-form{padding:16px;overflow-y:auto;}
.di-agg-form h5{margin:0 0 12px;font-size:1.05rem;font-weight:600;}
.di-agg-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:12px;}
.di-empty-ico{font-size:40px;color:var(--_di-border);margin-bottom:12px;display:block;}
.di-empty-text{color:var(--_di-muted);font-size:13px;padding:0 12px;}
[hidden]{display:none!important;}
`;
let V = !1;
function gt() {
  if (V || typeof document > "u") return;
  V = !0;
  const r = document.createElement("style");
  r.setAttribute("data-data-inspector", "1"), r.textContent = ht, document.head.appendChild(r);
}
class mt extends HTMLElement {
  constructor() {
    super(...arguments), this._pathInput = "", this._copied = !1, this._gridlookCopied = !1, this._activeTab = "metadata", this._dropdownOpen = !1, this._aggregationConfig = { ...ct }, this._output = null, this._built = !1, this._builtMode = !1, this._domOutput = void 0, this._domIframeUrl = void 0, this._domFileKey = void 0, this._copyTimer = null, this._gridlookTimer = null, this._restoreFocusTo = null, this._onOutsideClick = (t) => {
      const e = this.querySelector("#nc-dropdown-wrap");
      e && !e.contains(t.target) && this._dropdownOpen && (this._dropdownOpen = !1, this._syncDropdown());
    };
  }
  // ── Observed attributes ───────────────────────────────────────────────────
  static get observedAttributes() {
    return ["open", "file", "status", "error", "zarr-url", "zarr-status-code", "is-aggregation"];
  }
  // ── Attribute accessors ───────────────────────────────────────────────────
  get open() {
    return this.hasAttribute("open");
  }
  set open(t) {
    t ? this.setAttribute("open", "") : this.removeAttribute("open");
  }
  get file() {
    const t = this.getAttribute("file");
    if (!t) return null;
    try {
      return JSON.parse(t);
    } catch {
      return t;
    }
  }
  set file(t) {
    t === null ? this.removeAttribute("file") : this.setAttribute("file", Array.isArray(t) ? JSON.stringify(t) : t);
  }
  get status() {
    return this.getAttribute("status") ?? E.READY;
  }
  set status(t) {
    this.setAttribute("status", t);
  }
  /** Trusted xarray-repr HTML string. Setting triggers re-render. */
  get output() {
    return this._output;
  }
  set output(t) {
    this._output = t, this.isConnected && this._render();
  }
  get error() {
    return this.getAttribute("error");
  }
  set error(t) {
    t === null ? this.removeAttribute("error") : this.setAttribute("error", t);
  }
  get zarrUrl() {
    return this.getAttribute("zarr-url");
  }
  set zarrUrl(t) {
    t === null ? this.removeAttribute("zarr-url") : this.setAttribute("zarr-url", t);
  }
  get zarrStatusCode() {
    const t = this.getAttribute("zarr-status-code");
    return t === null ? null : parseInt(t, 10);
  }
  set zarrStatusCode(t) {
    t === null ? this.removeAttribute("zarr-status-code") : this.setAttribute("zarr-status-code", String(t));
  }
  get isAggregation() {
    return this.hasAttribute("is-aggregation");
  }
  set isAggregation(t) {
    t ? this.setAttribute("is-aggregation", "") : this.removeAttribute("is-aggregation");
  }
  // ── Lifecycle ─────────────────────────────────────────────────────────────
  connectedCallback() {
    const t = this.file;
    t && !Array.isArray(t) && (this._pathInput = t), this.open && (this._restoreFocusTo = document.activeElement), this._render(), document.addEventListener("mousedown", this._onOutsideClick);
  }
  disconnectedCallback() {
    document.removeEventListener("mousedown", this._onOutsideClick), this._clearTimers();
  }
  attributeChangedCallback(t, e, i) {
    var a, o;
    if (t === "open" && i !== null) {
      this._restoreFocusTo === null && (this._restoreFocusTo = document.activeElement), this._activeTab = "metadata", this._copied = !1, this._gridlookCopied = !1;
      const s = this.file;
      s && this._output === null && this.status === E.READY && !this.isAggregation && this._emit("inspector-submit", { file: s, aggregationConfig: null });
    }
    if (t === "file" && (i && !i.startsWith("[") && (this._pathInput = i), e !== null && e !== i && (this._output = null, this._domOutput = void 0, this._activeTab = "metadata", this._copied = !1, this._gridlookCopied = !1, this.hasAttribute("error") && this.removeAttribute("error"), this.hasAttribute("zarr-url") && this.removeAttribute("zarr-url"))), t === "status" && i === E.ERROR && (this._activeTab = "metadata"), t === "zarr-status-code" && this.isConnected && this._built) {
      const s = i ?? "3";
      (a = this.querySelector("#nc-pre-steps")) == null || a.setAttribute("status-code", s), (o = this.querySelector("#nc-body-steps")) == null || o.setAttribute("status-code", s);
      return;
    }
    this.isConnected && this._render();
  }
  // ── Event helpers ─────────────────────────────────────────────────────────
  _emit(t, e) {
    this.dispatchEvent(new CustomEvent(t, { bubbles: !0, composed: !0, detail: e }));
  }
  _clearTimers() {
    this._copyTimer !== null && (clearTimeout(this._copyTimer), this._copyTimer = null), this._gridlookTimer !== null && (clearTimeout(this._gridlookTimer), this._gridlookTimer = null);
  }
  // ── Action handlers ───────────────────────────────────────────────────────
  _handleInspect() {
    this.isAggregation ? this._emit("inspector-submit", {
      file: this.file,
      aggregationConfig: this._aggregationConfig
    }) : this._pathInput.trim() && this._emit("inspector-submit", { file: this._pathInput.trim(), aggregationConfig: null });
  }
  _handleInspectReload() {
    this._dropdownOpen = !1, this._syncDropdown(), this._pathInput.trim() && this._emit("inspector-submit", {
      file: this._pathInput.trim(),
      aggregationConfig: { reload: !0 }
    });
  }
  _copy(t, e) {
    ut(t).then(() => {
      e === "zarr" ? this._copied = !0 : this._gridlookCopied = !0, this.open && this._update();
      const a = setTimeout(() => {
        e === "zarr" ? this._copied = !1 : this._gridlookCopied = !1, this.isConnected && this.open && this._update();
      }, 2e3);
      e === "zarr" ? this._copyTimer = a : this._gridlookTimer = a;
    }).catch(() => {
    });
  }
  // ── Render orchestration ──────────────────────────────────────────────────
  _render() {
    if (!this.open) {
      this._teardown();
      return;
    }
    (!this._built || this._builtMode !== this.isAggregation) && this._build(), this._update();
  }
  _teardown() {
    this._clearTimers(), this.innerHTML = "", this._built = !1, this._domOutput = void 0, this._domIframeUrl = void 0, this._domFileKey = void 0;
    const t = this._restoreFocusTo;
    this._restoreFocusTo = null, t && t.isConnected && typeof t.focus == "function" && t.focus();
  }
  _q(t) {
    return this.querySelector(t);
  }
  _toggle(t, e) {
    t && (t.hidden = !e);
  }
  _setCopyBtn(t, e, i) {
    const a = this._q(t);
    a && (a.setAttribute("title", e ? "Copied!" : i), a.innerHTML = v(e ? m.check : m.copy));
  }
  // ── Build the static skeleton (once per open session / mode) ───────────────
  _build() {
    var n;
    gt();
    const t = this.isAggregation;
    this._builtMode = t, this._domOutput = void 0, this._domIframeUrl = void 0, this._domFileKey = void 0;
    const e = `
      <div class="di-header">
        <div class="di-header-row">
          <div class="di-header-main">
            <h1 class="di-title">
              <span class="di-title-ico">${v(t ? m.layers : m.info)}</span>
              <span id="nc-title">${t ? "Aggregate Files" : "File Inspector"}</span>
            </h1>
            ${t ? `<div id="nc-file-list-wrap" class="di-file-list" hidden>
                     <div id="nc-file-list-label" class="di-file-list-label"></div>
                     <ul id="nc-file-list"></ul>
                   </div>` : this._pathBarHtml()}
            <div id="nc-zarr-row" class="di-zarr-row" hidden>
              <div class="di-zarr-inner">
                <span class="di-muted" style="display:inline-flex;flex-shrink:0">${v(m.link)}</span>
                <span class="di-muted" style="font-weight:500;flex-shrink:0">Zarr:</span>
                <code id="nc-zarr-url" class="di-code"></code>
                <button id="nc-copy-zarr" class="di-btn di-btn-outline" title="Copy Zarr URL">${v(m.copy)}</button>
              </div>
            </div>
          </div>
          <button id="nc-close-btn" class="di-close" aria-label="Close">&times;</button>
        </div>
      </div>`, i = t ? `<div id="nc-agg-form" class="di-agg-form" hidden>
           <h5>Aggregation Configuration</h5>
           <aggregation-config id="nc-agg-config"></aggregation-config>
           <div class="di-agg-actions">
             <button id="nc-cancel-btn" class="di-btn di-btn-secondary">Cancel</button>
             <button id="nc-aggregate-btn" class="di-btn di-btn-primary">${v(m.compress, !0)}Aggregate Files</button>
           </div>
         </div>` : "", a = `
      <div id="nc-pre-loading" class="di-center" hidden>
        <zarr-loading-steps id="nc-pre-steps" status-code="3"${t ? " is-aggregation" : ""}></zarr-loading-steps>
        <p id="nc-pre-loading-text" class="di-muted" style="margin-top:12px;font-size:13px;"></p>
      </div>`, o = `
      <div id="nc-tabs-wrap" hidden>
        <div class="di-tabs" role="tablist" aria-label="Inspector views">
          <button id="nc-tab-metadata" data-tab="metadata" class="nc-tab-btn di-tab" role="tab" aria-controls="nc-metadata" aria-selected="true">${v(m.database, !0)}Metadata</button>
          <button id="nc-tab-gridlook" data-tab="gridlook" class="nc-tab-btn di-tab" role="tab" aria-controls="nc-gridlook" aria-selected="false">${v(m.cube, !0)}3D Viewer</button>
        </div>
        <div id="nc-body" class="di-body">
          <div id="nc-error" class="di-error" hidden>
            <div class="di-error-row">
              <span class="di-error-ico">${v(m.alert)}</span>
              <div class="di-error-body">
                <strong class="di-error-title">Error loading metadata</strong>
                <div id="nc-error-msg" class="di-error-msg"></div>
                <button id="nc-retry-btn" class="di-btn di-btn-danger">${v(m.refresh, !0)}Retry</button>
              </div>
            </div>
          </div>

          <div id="nc-loading" class="di-center" hidden>
            <zarr-loading-steps id="nc-body-steps" status-code="3"${t ? " is-aggregation" : ""}></zarr-loading-steps>
            <p id="nc-loading-text" class="di-muted" style="margin-top:12px;font-size:13px;"></p>
          </div>

          <div id="nc-metadata" class="di-metadata" role="tabpanel" aria-labelledby="nc-tab-metadata" tabindex="0" hidden>
            <div id="nc-metadata-inner"></div>
          </div>

          <div id="nc-gridlook" role="tabpanel" aria-labelledby="nc-tab-gridlook" tabindex="0" hidden>
            <div class="di-gridlook-bar">
              <div class="di-gridlook-inner">
                <span class="di-gridlook-ico">${v(m.external)}</span>
                <span class="di-gridlook-label">GridLook URL:</span>
                <code id="nc-gridlook-url" class="di-gridlook-code"></code>
                <button id="nc-copy-gridlook" class="di-btn di-gridlook-btn" title="Copy link">${v(m.copy)}</button>
                <button id="nc-refresh-gridlook" class="di-btn di-gridlook-btn" title="Refresh GridLook viewer">${v(m.refresh)}</button>
                <button id="nc-open-gridlook" class="di-btn di-gridlook-btn-primary" title="Open in new tab">${v(m.external, !0)}Open in New Tab</button>
              </div>
            </div>
            <div id="nc-gridlook-frame" class="di-gridlook-frame"></div>
          </div>

          <div id="nc-empty-body" class="di-center" hidden>
            <span class="di-empty-ico">${v(m.database)}</span>
            <p id="nc-empty-body-text" class="di-empty-text"></p>
          </div>
        </div>
      </div>`, s = `
      <div id="nc-empty-main" class="di-center" hidden>
        <span class="di-empty-ico">${v(m.database)}</span>
        <p id="nc-empty-main-text" class="di-empty-text"></p>
      </div>`;
    this.innerHTML = `
      <div id="nc-backdrop" class="di-backdrop">
        <div class="di-modal" role="dialog" aria-modal="true" aria-labelledby="nc-title" tabindex="-1">
          ${e}
          ${i}
          ${a}
          ${o}
          ${s}
        </div>
      </div>`, (n = this._q("#nc-agg-config")) == null || n.setAttribute(
      "initial-config",
      JSON.stringify(this._aggregationConfig)
    ), this._built = !0, this._attach(), this._initialFocus();
  }
  _ensureIframe() {
    const t = this._q("#nc-gridlook-iframe");
    if (t) return t;
    const e = this._q("#nc-gridlook-frame");
    if (!e) return null;
    const i = document.createElement("iframe");
    i.id = "nc-gridlook-iframe", i.title = "GridLook 3D Viewer", i.setAttribute("sandbox", "allow-scripts allow-same-origin allow-popups allow-downloads"), i.setAttribute("referrerpolicy", "no-referrer"), i.setAttribute("loading", "lazy");
    try {
      e.appendChild(i);
    } catch {
    }
    return i;
  }
  /** Assign an iframe src defensively (never let a load error abort a render). */
  _setIframeSrc(t, e) {
    try {
      t.src = e;
    } catch {
    }
  }
  _pathBarHtml() {
    return `
      <div class="di-pathbar">
        <label class="di-pathbar-label" for="nc-path-input">File path:</label>
        <div class="di-pathbar-row">
          <input id="nc-path-input" type="text" class="di-input" placeholder="/path/to/data.nc" />
          <div id="nc-dropdown-wrap" class="di-dropdown-wrap di-btn-group">
            <button id="nc-load-btn" class="di-btn di-btn-primary">${v(m.load, !0)}Load</button>
            <button id="nc-load-toggle" class="di-btn di-btn-primary di-btn-split" title="More load options">${v(m.chevronDown)}</button>
            <ul id="nc-dropdown-menu" class="di-menu" hidden>
              <li>
                <button id="nc-reload-btn" class="di-menu-item">${v(m.ban, !0)}Force Reload (bypass cache)</button>
              </li>
            </ul>
          </div>
        </div>
      </div>`;
  }
  // ── Attach listeners once (nodes persist across updates) ───────────────────
  _attach() {
    var e, i, a, o, s, n, l, d, h, p, u, y, _, $;
    (e = this._q("#nc-backdrop")) == null || e.addEventListener("click", (c) => {
      c.target === c.currentTarget && this._emit("inspector-close", null);
    }), (i = this._q("#nc-backdrop")) == null || i.addEventListener(
      "keydown",
      (c) => this._onKeydown(c)
    ), (a = this._q("#nc-close-btn")) == null || a.addEventListener("click", () => this._emit("inspector-close", null)), (o = this._q("#nc-cancel-btn")) == null || o.addEventListener("click", () => this._emit("inspector-close", null)), (s = this._q("#nc-load-btn")) == null || s.addEventListener("click", () => this._handleInspect()), (n = this._q("#nc-aggregate-btn")) == null || n.addEventListener("click", () => this._handleInspect()), (l = this._q("#nc-retry-btn")) == null || l.addEventListener("click", () => this._handleInspect());
    const t = this._q("#nc-path-input");
    t && (t.value = this._pathInput, t.addEventListener("input", (c) => {
      this._pathInput = c.target.value;
    }), t.addEventListener("keypress", (c) => {
      c.key === "Enter" && this._handleInspect();
    })), (d = this._q("#nc-load-toggle")) == null || d.addEventListener("click", () => {
      this._dropdownOpen = !this._dropdownOpen, this._syncDropdown();
    }), (h = this._q("#nc-reload-btn")) == null || h.addEventListener("click", () => this._handleInspectReload()), (p = this._q("#nc-copy-zarr")) == null || p.addEventListener("click", () => {
      const c = this.zarrUrl;
      c && this._copy(c, "zarr");
    }), (u = this._q("#nc-copy-gridlook")) == null || u.addEventListener("click", () => {
      const c = this.zarrUrl;
      c && this._copy(H(c), "gridlook");
    }), (y = this._q("#nc-refresh-gridlook")) == null || y.addEventListener("click", () => {
      var c;
      this.zarrUrl && ((c = this._q("#nc-gridlook-iframe")) == null || c.remove(), this._domIframeUrl = void 0, this._update());
    }), (_ = this._q("#nc-open-gridlook")) == null || _.addEventListener("click", () => {
      const c = this.zarrUrl;
      c && window.open(H(c), "_blank", "noopener,noreferrer");
    }), this.querySelectorAll(".nc-tab-btn").forEach((c) => {
      c.addEventListener("click", () => {
        const k = c.dataset.tab;
        k && !c.disabled && (this._activeTab = k, this._update());
      });
    }), ($ = this._q("#nc-agg-config")) == null || $.addEventListener("config-change", (c) => {
      this._aggregationConfig = c.detail;
    });
  }
  // ── Patch dynamic state (no DOM rebuild) ───────────────────────────────────
  _update() {
    var j, O;
    const t = this.isAggregation, e = this.status, i = e === E.LOADING, a = e === E.READY, o = e === E.ERROR, s = this.zarrUrl, n = this.file, l = Array.isArray(n) ? n : [], d = this._output != null && this._output !== "", h = String(this.zarrStatusCode ?? 3), p = t ? "Aggregating files and loading metadata..." : "Loading metadata...", u = t ? "Configure aggregation settings and click 'Aggregate Files' to begin" : "Enter a file path and click Load to inspect metadata";
    if (t) {
      const b = this._q("#nc-file-list-wrap"), A = l.length > 0;
      if (this._toggle(b, A), A) {
        const I = this._q("#nc-file-list-label");
        I && (I.textContent = `Selected files (${l.length}):`);
        const K = this._q("#nc-file-list"), W = l.join("\0");
        K && this._domFileKey !== W && (K.replaceChildren(
          ...l.map((lt) => {
            const P = document.createElement("li");
            return P.textContent = lt, P;
          })
        ), this._domFileKey = W);
      }
    }
    if (!t) {
      const b = this._q("#nc-load-btn"), A = this._q("#nc-load-toggle");
      b && (b.disabled = i), A && (A.disabled = i), this._syncDropdown();
    }
    const _ = !!s && s !== (typeof n == "string" ? n : null);
    if (this._toggle(this._q("#nc-zarr-row"), _), _ && s) {
      const b = this._q("#nc-zarr-url");
      b && (b.textContent = s);
    }
    this._setCopyBtn("#nc-copy-zarr", this._copied, "Copy Zarr URL"), t && this._toggle(this._q("#nc-agg-form"), a && !d), this._toggle(this._q("#nc-pre-loading"), !s && i), this._toggle(this._q("#nc-tabs-wrap"), !!s && (d || i || o)), this._toggle(this._q("#nc-empty-main"), !s && !i && a && !d);
    const $ = this._q("#nc-pre-loading-text");
    $ && ($.textContent = p), (j = this._q("#nc-pre-steps")) == null || j.setAttribute("status-code", h);
    const c = this._q('[data-tab="metadata"]'), k = this._q('[data-tab="gridlook"]');
    if (c) {
      const b = this._activeTab === "metadata";
      c.classList.toggle("di-tab-active", b), c.setAttribute("aria-selected", String(b));
    }
    if (k) {
      const b = this._activeTab === "gridlook";
      k.disabled = e !== E.READY || !d, k.classList.toggle("di-tab-active", b), k.setAttribute("aria-selected", String(b));
    }
    this._toggle(this._q("#nc-error"), o && this._activeTab === "metadata");
    const g = this._q("#nc-error-msg");
    g && (g.textContent = this.error ?? ""), this._toggle(this._q("#nc-loading"), i);
    const f = this._q("#nc-loading-text");
    f && (f.textContent = p), (O = this._q("#nc-body-steps")) == null || O.setAttribute("status-code", h), this._toggle(
      this._q("#nc-metadata"),
      this._activeTab === "metadata" && d && !i && !o
    );
    const w = this._q("#nc-metadata-inner");
    w && this._domOutput !== this._output && (w.innerHTML = this._output ?? "", this._domOutput = this._output);
    const C = this._activeTab === "gridlook" && !!s;
    if (this._toggle(this._q("#nc-gridlook"), C), s) {
      const b = H(s), A = this._q("#nc-gridlook-url");
      if (A && (A.textContent = b), C) {
        const I = this._ensureIframe();
        I && this._domIframeUrl !== b && (this._setIframeSrc(I, b), this._domIframeUrl = b);
      }
    }
    this._setCopyBtn("#nc-copy-gridlook", this._gridlookCopied, "Copy link"), this._toggle(this._q("#nc-empty-body"), !d && !i && !o);
    const M = this._q("#nc-empty-body-text");
    M && (M.textContent = u);
    const S = this._q("#nc-empty-main-text");
    S && (S.textContent = u);
  }
  _syncDropdown() {
    const t = this._q("#nc-dropdown-menu");
    t && (t.hidden = !this._dropdownOpen);
  }
  // ── Accessibility: focus trap, initial focus, Escape-to-close ──────────────-
  _focusables() {
    const t = this._q(".di-modal");
    return t ? Array.from(t.querySelectorAll('a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])')).filter(
      (i) => !i.closest("[hidden]")
    ) : [];
  }
  _initialFocus() {
    var e;
    const t = this._q("#nc-path-input") ?? this._focusables()[0] ?? this._q(".di-modal");
    (e = t == null ? void 0 : t.focus) == null || e.call(t);
  }
  _onKeydown(t) {
    if (t.key === "Escape") {
      t.stopPropagation(), this._emit("inspector-close", null);
      return;
    }
    if (t.key !== "Tab") return;
    const e = this._focusables();
    if (e.length === 0) {
      t.preventDefault();
      return;
    }
    const i = e[0], a = e[e.length - 1], o = document.activeElement;
    t.shiftKey && (o === i || !this.contains(o)) ? (t.preventDefault(), a.focus()) : !t.shiftKey && o === a && (t.preventDefault(), i.focus());
  }
}
customElements.define("data-inspector", mt);
const Z = `<svg viewBox="0 0 448 512" width="10" height="10" style="vertical-align:middle;">
  <path d="M207.029 381.476L12.686 187.132c-9.373-9.373-9.373-24.569 0-33.941l22.667-22.667
    c9.357-9.357 24.522-9.375 33.901-.04L224 284.505l154.745-154.021c9.379-9.335
    24.544-9.317 33.901.04l22.667 22.667c9.373 9.373 9.373 24.569 0 33.941
    L240.971 381.476c-9.373 9.372-24.569 9.372-33.942 0z" fill="currentColor"/>
</svg>`, Y = `<svg viewBox="0 0 256 512" width="6" height="10" style="vertical-align:middle;">
  <path d="M224.3 273l-136 136c-9.4 9.4-24.6 9.4-33.9 0l-22.6-22.6c-9.4-9.4-9.4-24.6
    0-33.9l96.4-96.4-96.4-96.4c-9.4-9.4-9.4-24.6 0-33.9L54.3 103c9.4-9.4
    24.6-9.4 33.9 0l136 136c9.5 9.4 9.5 24.6.1 34z" fill="currentColor"/>
</svg>`, F = {
  aggregate: "auto",
  join: null,
  compat: null,
  data_vars: null,
  coords: null,
  dim: "",
  group_by: "",
  reload: !1,
  access_pattern: "map",
  chunk_size: 16,
  map_primary_chunksize: 1,
  timeout: 120
};
class ft extends HTMLElement {
  constructor() {
    super(...arguments), this._config = { ...F }, this._showAdvanced = !1, this._handleChange = (t) => {
      const e = t.target, i = e.dataset.field;
      if (!i) return;
      let a;
      if (e.type === "checkbox")
        a = e.checked;
      else if (e.type === "number") {
        const o = parseFloat(e.value);
        a = Number.isFinite(o) ? o : parseInt(e.value, 10);
      } else
        a = e.value === "" ? null : e.value;
      this._config = { ...this._config, [i]: a }, this._emitChange(), this._updateConditionals();
    }, this._handleClick = (t) => {
      if (t.target.closest("#nc-advanced-toggle")) {
        this._showAdvanced = !this._showAdvanced;
        const e = this.querySelector(".nc-advanced"), i = this.querySelector(".nc-chevron");
        e && (e.style.display = this._showAdvanced ? "block" : "none"), i && (i.innerHTML = this._showAdvanced ? Z : Y);
      }
    };
  }
  connectedCallback() {
    try {
      const t = JSON.parse(
        this.getAttribute("initial-config") ?? "{}"
      );
      this._config = { ...F, ...t };
    } catch {
      this._config = { ...F };
    }
    this._render(), this.addEventListener("change", this._handleChange), this.addEventListener("click", this._handleClick);
  }
  disconnectedCallback() {
    this.removeEventListener("change", this._handleChange), this.removeEventListener("click", this._handleClick);
  }
  _emitChange() {
    this.dispatchEvent(
      new CustomEvent("config-change", {
        bubbles: !0,
        composed: !0,
        detail: this._config
      })
    );
  }
  /** Show/hide conditional fields without re-rendering. */
  _updateConditionals() {
    const t = this.querySelector(".nc-dim-field"), e = this.querySelector(".nc-map-chunksize");
    t && (t.style.display = this._config.aggregate === "concat" ? "block" : "none"), e && (e.style.display = this._config.access_pattern === "map" ? "block" : "none");
  }
  _render() {
    const t = this._config;
    this.innerHTML = `
      <div class="aggregation-config">

        <!-- Aggregation Method -->
        <div class="mb-3">
          <label class="form-label fw-semibold">Aggregation Method</label>
          <select class="form-select form-select-sm" data-field="aggregate">
            <option value="auto"   ${t.aggregate === "auto" ? "selected" : ""}>Auto (Detect automatically)</option>
            <option value="merge"  ${t.aggregate === "merge" ? "selected" : ""}>Merge (Combine variables)</option>
            <option value="concat" ${t.aggregate === "concat" ? "selected" : ""}>Concat (Join along dimension)</option>
          </select>
          <div class="form-text text-muted">Auto mode will automatically detect the best aggregation method</div>
        </div>

        <!-- Timeout -->
        <div class="mb-3">
          <label class="form-label fw-semibold">Timeout (seconds)</label>
          <input type="number" class="form-control form-control-sm"
            data-field="timeout" min="10" max="3600" value="${t.timeout ?? 120}">
          <div class="form-text text-muted">
            Max wait time for the aggregation to complete (default: 120 s). Increase for large datasets.
          </div>
        </div>

        <!-- Advanced Toggle -->
        <button type="button" id="nc-advanced-toggle"
          class="btn btn-link btn-sm p-0 mb-3 text-decoration-none">
          <span class="nc-chevron">${this._showAdvanced ? Z : Y}</span>
          <span class="ms-2">Advanced Options</span>
        </button>

        <!-- Advanced Options -->
        <div class="nc-advanced" style="display:${this._showAdvanced ? "block" : "none"};">

          <!-- Dimension (concat only) -->
          <div class="mb-3 nc-dim-field" style="display:${t.aggregate === "concat" ? "block" : "none"};">
            <label class="form-label">Dimension to Concatenate Along</label>
            <input type="text" class="form-control form-control-sm"
              data-field="dim" placeholder="e.g., time, ensemble" value="${t.dim ?? ""}">
            <div class="form-text text-muted">Leave empty to create a new dimension</div>
          </div>

          <!-- Join Mode -->
          <div class="mb-3">
            <label class="form-label">Join Mode</label>
            <select class="form-select form-select-sm" data-field="join">
              <option value=""      ${t.join ? "" : "selected"}>Default</option>
              <option value="outer" ${t.join === "outer" ? "selected" : ""}>Outer (Union)</option>
              <option value="inner" ${t.join === "inner" ? "selected" : ""}>Inner (Intersection)</option>
              <option value="left"  ${t.join === "left" ? "selected" : ""}>Left</option>
              <option value="right" ${t.join === "right" ? "selected" : ""}>Right</option>
              <option value="exact" ${t.join === "exact" ? "selected" : ""}>Exact (Must match)</option>
            </select>
          </div>

          <!-- Compatibility Mode -->
          <div class="mb-3">
            <label class="form-label">Compatibility Mode</label>
            <select class="form-select form-select-sm" data-field="compat">
              <option value=""              ${t.compat ? "" : "selected"}>Default</option>
              <option value="no_conflicts"  ${t.compat === "no_conflicts" ? "selected" : ""}>No Conflicts</option>
              <option value="equals"        ${t.compat === "equals" ? "selected" : ""}>Equals</option>
              <option value="override"      ${t.compat === "override" ? "selected" : ""}>Override</option>
            </select>
          </div>

          <!-- Data Variables -->
          <div class="mb-3">
            <label class="form-label">Data Variables Handling</label>
            <select class="form-select form-select-sm" data-field="data_vars">
              <option value=""          ${t.data_vars ? "" : "selected"}>Default</option>
              <option value="minimal"   ${t.data_vars === "minimal" ? "selected" : ""}>Minimal</option>
              <option value="different" ${t.data_vars === "different" ? "selected" : ""}>Different</option>
              <option value="all"       ${t.data_vars === "all" ? "selected" : ""}>All</option>
            </select>
          </div>

          <!-- Coordinates -->
          <div class="mb-3">
            <label class="form-label">Coordinates Handling</label>
            <select class="form-select form-select-sm" data-field="coords">
              <option value=""          ${t.coords ? "" : "selected"}>Default</option>
              <option value="minimal"   ${t.coords === "minimal" ? "selected" : ""}>Minimal</option>
              <option value="different" ${t.coords === "different" ? "selected" : ""}>Different</option>
              <option value="all"       ${t.coords === "all" ? "selected" : ""}>All</option>
            </select>
          </div>

          <!-- Group By -->
          <div class="mb-3">
            <label class="form-label">Group By (Optional)</label>
            <input type="text" class="form-control form-control-sm"
              data-field="group_by" placeholder="e.g., ensemble, variable" value="${t.group_by ?? ""}">
            <div class="form-text text-muted">Group files by a specific attribute</div>
          </div>

          <!-- Reload Cache -->
          <div class="mb-3">
            <div class="form-check">
              <input type="checkbox" class="form-check-input" id="aggregation-reload"
                data-field="reload" ${t.reload ? "checked" : ""}>
              <label class="form-check-label" for="aggregation-reload">Force Reload (bypass cache)</label>
            </div>
            <div class="form-text text-muted">
              Force server to fetch fresh data instead of using cached version
            </div>
          </div>

          <!-- Access Pattern -->
          <div class="mb-3">
            <label class="form-label">Access Pattern Optimization</label>
            <select class="form-select form-select-sm" data-field="access_pattern">
              <option value="map"         ${t.access_pattern === "map" ? "selected" : ""}>Map (spatial slices)</option>
              <option value="time_series" ${t.access_pattern === "time_series" ? "selected" : ""}>Time Series (temporal slices)</option>
            </select>
            <div class="form-text text-muted">Optimize chunk layout for your typical data access pattern</div>
          </div>

          <!-- Chunk Size -->
          <div class="mb-3">
            <label class="form-label">Target Chunk Size (MB)</label>
            <input type="number" class="form-control form-control-sm"
              data-field="chunk_size" step="0.1" min="1" max="1000" value="${t.chunk_size ?? 16}">
            <div class="form-text text-muted">Target size for data chunks (default: 16 MB)</div>
          </div>

          <!-- Map Primary Chunksize (map pattern only) -->
          <div class="mb-3 nc-map-chunksize" style="display:${t.access_pattern === "map" ? "block" : "none"};">
            <label class="form-label">Primary Dimension Chunk Size</label>
            <input type="number" class="form-control form-control-sm"
              data-field="map_primary_chunksize" min="1" value="${t.map_primary_chunksize ?? 1}">
            <div class="form-text text-muted">Number of time steps per chunk (for map access pattern)</div>
          </div>

        </div>
      </div>`;
  }
}
customElements.define("aggregation-config", ft);
const bt = `
  @keyframes zarrPulseRing {
    0%   { transform: scale(0.9); opacity: 0.7; }
    60%  { transform: scale(1.6); opacity: 0;   }
    100% { transform: scale(0.9); opacity: 0;   }
  }
  @keyframes zarrSpinArc {
    from { transform: rotate(0deg);   }
    to   { transform: rotate(360deg); }
  }
  @keyframes zarrMsgIn {
    0%   { opacity: 0; transform: translateY(5px);  }
    18%  { opacity: 1; transform: translateY(0);     }
    82%  { opacity: 1; transform: translateY(0);     }
    100% { opacity: 0; transform: translateY(-5px);  }
  }
  @keyframes zarrTrackFill {
    from { width: 0%; }
    to   { width: 100%; }
  }
  .zarr-spin { animation: zarrSpinArc 1.3s linear infinite; transform-origin: center; }
  .zarr-msg  { animation: zarrMsgIn 2.8s ease-in-out forwards; }
`, J = [
  { id: "submitted", label: "Submitted" },
  { id: "queued", label: "Queued" },
  { id: "converting", label: "Converting" },
  { id: "ready", label: "Ready" }
], x = {
  done: "#0d9488",
  active: "#14b8a6",
  pending: "#d1d5db",
  track: "#e5e7eb",
  textOn: "#0f766e",
  textOff: "#9ca3af",
  msg: "#6b7280",
  timer: "#d1d5db"
}, L = [
  "Reading file structure…",
  "Analysing coordinate metadata…",
  "Optimising chunk layout…",
  "Building Zarr metadata store…",
  "Assembling data variables…",
  "Applying access-pattern optimisation…",
  "Finalising dataset…"
], q = [
  "Aligning coordinate indexes…",
  "Resolving variable conflicts…",
  "Concatenating along dimension…",
  "Merging datasets…",
  "Validating compatibility…",
  "Assembling aggregated store…",
  "Almost there…"
];
function D(r) {
  return r === 0 ? 3 : r === 4 ? 2 : r === 3 ? 1 : 0;
}
function vt() {
  if (!document.getElementById("zarr-loading-keyframes")) {
    const r = document.createElement("style");
    r.id = "zarr-loading-keyframes", r.textContent = bt, document.head.appendChild(r);
  }
}
function X(r, t, e) {
  const i = r < t, a = r === t, o = i ? `<svg width="10" height="8" viewBox="0 0 10 8" fill="none">
        <path d="M1 4L3.8 7L9 1" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
       </svg>` : a ? `<svg width="16" height="16" viewBox="0 0 16 16" class="zarr-spin" style="position:absolute;">
        <circle cx="8" cy="8" r="5.5" fill="none" stroke="${x.active}" stroke-width="2"
          stroke-dasharray="18 17" stroke-linecap="round"/>
       </svg>` : `<div style="width:5px;height:5px;border-radius:50%;background:${x.pending};"></div>`, s = a ? `<div style="position:absolute;width:30px;height:30px;border-radius:50%;
         border:2px solid ${x.active};animation:zarrPulseRing 2s ease-out infinite;pointer-events:none;"></div>` : "";
  return `
    ${r > 0 ? `
    <div style="flex:1;height:2px;background:${i ? `linear-gradient(90deg,${x.done},${x.active})` : x.track};
      transition:background 0.5s ease;position:relative;overflow:hidden;">
      ${a ? `<div style="position:absolute;top:0;left:0;height:100%;
        background:linear-gradient(90deg,${x.done},${x.active});
        animation:zarrTrackFill 0.6s ease forwards;"></div>` : ""}
    </div>` : ""}
    <div style="position:relative;display:flex;flex-direction:column;align-items:center;">
      ${s}
      <div style="width:22px;height:22px;border-radius:50%;
        background:${i ? x.done : a ? "#fff" : "#f9fafb"};
        border:2px solid ${i ? x.done : a ? x.active : x.pending};
        display:flex;align-items:center;justify-content:center;
        position:relative;z-index:1;
        box-shadow:${a ? "0 0 0 4px rgba(20,184,166,0.12)" : "none"};
        transition:all 0.4s ease;">
        ${o}
      </div>
      <span style="position:absolute;top:28px;font-size:10.5px;
        font-weight:${a ? 600 : 400};
        color:${i ? x.done : a ? x.textOn : x.textOff};
        white-space:nowrap;letter-spacing:0.03em;text-transform:uppercase;
        transition:color 0.4s ease;">
        ${e}
      </span>
    </div>`;
}
class xt extends HTMLElement {
  constructor() {
    super(...arguments), this._msgIdx = 0, this._msgKey = 0, this._elapsed = 0, this._startTime = Date.now(), this._msgTimer = null, this._elapsedTimer = null;
  }
  static get observedAttributes() {
    return ["status-code", "is-aggregation"];
  }
  get statusCode() {
    return parseInt(this.getAttribute("status-code") ?? "3", 10);
  }
  get isAggregation() {
    return this.hasAttribute("is-aggregation");
  }
  connectedCallback() {
    vt(), this._render(), this._startTimers();
  }
  disconnectedCallback() {
    this._stopTimers();
  }
  attributeChangedCallback() {
    this.isConnected && this._updateStages();
  }
  _startTimers() {
    this._startTime = Date.now(), this._elapsedTimer = setInterval(() => {
      this._elapsed = Math.floor((Date.now() - this._startTime) / 1e3);
      const t = this.querySelector(".zarr-elapsed");
      t && (t.textContent = this._formatElapsed(this._elapsed));
    }, 1e3), this._startMsgCycle();
  }
  _startMsgCycle() {
    if (this._msgTimer && clearInterval(this._msgTimer), D(this.statusCode) !== 2) return;
    const e = this.isAggregation ? q : L;
    this._msgTimer = setInterval(() => {
      this._msgIdx = (this._msgIdx + 1) % e.length, this._msgKey += 1, this._updateMessage();
    }, 2800);
  }
  _stopTimers() {
    this._msgTimer && (clearInterval(this._msgTimer), this._msgTimer = null), this._elapsedTimer && (clearInterval(this._elapsedTimer), this._elapsedTimer = null);
  }
  _formatElapsed(t) {
    return t < 60 ? `${t}s` : `${Math.floor(t / 60)}m ${t % 60}s`;
  }
  _render() {
    const t = D(this.statusCode), e = this.isAggregation ? q : L;
    this.innerHTML = `
      <div style="padding:28px 12px 20px;display:flex;flex-direction:column;align-items:center;gap:0;">
        <div class="zarr-stages" style="display:flex;align-items:center;width:100%;max-width:360px;margin-bottom:28px;">
          ${J.map((i, a) => X(a, t, i.label)).join("")}
        </div>

        <div style="height:20px;"></div>

        <div class="zarr-msg-container" style="min-height:22px;text-align:center;">
          ${this._renderMessage(t, e)}
        </div>

        <div style="margin-top:10px;">
          <span class="zarr-elapsed" style="font-size:11px;color:${x.timer};
            font-variant-numeric:tabular-nums;letter-spacing:0.05em;">
            ${this._formatElapsed(this._elapsed)}
          </span>
        </div>
      </div>`;
  }
  _renderMessage(t, e) {
    return t === 2 ? `<span key="${this._msgKey}" class="zarr-msg" style="font-size:13px;color:${x.msg};">
                ${e[this._msgIdx]}
              </span>` : t === 1 ? `<span style="font-size:13px;color:${x.msg};">Waiting for a worker to pick up the task…</span>` : "";
  }
  /** Update only the stage dots (called when statusCode changes). */
  _updateStages() {
    const t = D(this.statusCode), e = this.querySelector(".zarr-stages");
    e && (e.innerHTML = J.map((a, o) => X(o, t, a.label)).join(""));
    const i = this.querySelector(".zarr-msg-container");
    if (i) {
      const a = this.isAggregation ? q : L;
      i.innerHTML = this._renderMessage(t, a);
    }
    this._startMsgCycle();
  }
  /** Update only the rotating message text. */
  _updateMessage() {
    const t = D(this.statusCode), e = this.querySelector(".zarr-msg-container");
    if (e) {
      const i = this.isAggregation ? q : L;
      e.innerHTML = this._renderMessage(t, i);
    }
  }
}
customElements.define("zarr-loading-steps", xt);
const yt = "freva_auth_token=";
function G() {
  if (typeof document > "u") return {};
  const t = document.cookie.split(";").find((e) => e.trim().startsWith(yt));
  if (!t) return {};
  try {
    let e = t.substring(t.indexOf("=") + 1).trim();
    return e.startsWith('"') && e.endsWith('"') && (e = e.slice(1, -1)), e ? { Authorization: `Bearer ${e}` } : {};
  } catch {
    return {};
  }
}
function st(r) {
  if (typeof r != "string") return r;
  let t = r.trim();
  for (; /^https?%(25)*3a/i.test(t); )
    try {
      const e = decodeURIComponent(t);
      if (e === t) break;
      t = e;
    } catch {
      break;
    }
  return t;
}
function _t(r) {
  return `/api/freva-nextgen/data-portal/zarr-utils/status?url=${r}&timeout=1`;
}
class qt {
  constructor(t, e = {}) {
    this.timer = null, this.cancelled = !1, this.zarrUrl = t, this.intervalMs = e.intervalMs ?? 2e3, this.getAuthHeaders = e.getAuthHeaders ?? G, this.getStatusUrl = e.getStatusUrl ?? _t, this.onStatus = e.onStatus ?? (() => {
    }), this.onError = e.onError ?? (() => {
    });
  }
  start() {
    this.cancelled = !1, this.poll(), this.timer = setInterval(() => {
      this.poll();
    }, this.intervalMs);
  }
  stop() {
    this.cancelled = !0, this.timer !== null && (clearInterval(this.timer), this.timer = null);
  }
  async poll() {
    try {
      const t = this.getStatusUrl(encodeURIComponent(this.zarrUrl)), e = await fetch(t, {
        credentials: "same-origin",
        headers: this.getAuthHeaders()
      });
      if (!e.ok) {
        this.cancelled || this.onStatus(5, null);
        return;
      }
      const i = await e.json(), a = i.status ?? 5, o = i.reason ?? null;
      this.cancelled || (this.onStatus(a, o), a <= 2 && this.stop());
    } catch (t) {
      this.cancelled || this.onError(t instanceof Error ? t.message : String(t));
    }
  }
}
const Q = { isZarr: !1, version: null, consolidated: !1 };
async function Dt(r, t = {}) {
  if (!r) return { ...Q };
  const e = st(r).replace(/\/$/, ""), a = {
    credentials: "same-origin",
    headers: (t.getAuthHeaders ?? G)(),
    signal: AbortSignal.timeout(t.timeoutMs ?? 5e3)
  };
  try {
    if ((await fetch(`${e}/.zmetadata`, a)).ok) return { isZarr: !0, version: 2, consolidated: !0 };
  } catch {
  }
  try {
    const o = await fetch(`${e}/zarr.json`, a);
    if (o.ok) {
      const s = await o.json(), n = s.zarr_format;
      if (n === 2 || n === 3) {
        const l = n === 3 ? "consolidated_metadata" in s : !1;
        return { isZarr: !0, version: n, consolidated: l };
      }
    }
  } catch {
  }
  return { ...Q };
}
function kt(r) {
  return {
    f2: "float16",
    f4: "float32",
    f8: "float64",
    i1: "int8",
    i2: "int16",
    i4: "int32",
    i8: "int64",
    u1: "uint8",
    u2: "uint16",
    u4: "uint32",
    u8: "uint64",
    b1: "bool"
  }[r.slice(1)] ?? r;
}
function wt(r) {
  const t = { ...r };
  return delete t._ARRAY_DIMENSIONS, t;
}
function nt(r, t, e) {
  const i = {}, a = {};
  for (const [d, h] of Object.entries(r)) {
    const { shape: p, chunks: u, dtype: y, dims: _, attrs: $ } = e(h);
    _.forEach((g, f) => {
      g in i || (i[g] = p[f] ?? 0);
    });
    const c = $.units, k = _.length === 1 && _[0] === d && (String(c ?? "").includes("since") || d === "time");
    a[d] = {
      shape: p,
      chunks: u,
      dtype: y,
      dims: _,
      attrs: $,
      _isTimeCoord: k
    };
  }
  const o = /* @__PURE__ */ new Set(), s = (d) => String(d ?? "").split(/[\s,]+/).filter(Boolean);
  for (const [d, h] of Object.entries(a))
    h.dims.length === 1 && h.dims[0] === d && o.add(d);
  s(t.coordinates).forEach((d) => o.add(d));
  for (const d of Object.values(a))
    s(d.attrs.coordinates).forEach((h) => o.add(h));
  const n = {}, l = {};
  for (const [d, h] of Object.entries(a))
    (o.has(d) ? n : l)[d] = h;
  return { dims: i, coords: n, data_vars: l, attrs: t };
}
function T(r) {
  return Object.keys(r.coords).length + Object.keys(r.data_vars).length > 0;
}
function B(r, t) {
  const e = r[`${t}.zattrs`] ?? {}, i = {};
  for (const [a, o] of Object.entries(r)) {
    if (!a.startsWith(t)) continue;
    const s = a.slice(t.length);
    if (s.endsWith("/.zarray")) {
      const n = s.slice(0, -8);
      if (n.includes("/")) continue;
      i[n] ?? (i[n] = {}), i[n].zarray = o;
    } else if (s.endsWith("/.zattrs")) {
      const n = s.slice(0, -8);
      if (!n || n.includes("/")) continue;
      i[n] ?? (i[n] = {}), i[n].zattrs = o;
    }
  }
  for (const a of Object.keys(i))
    i[a].zarray || delete i[a];
  return nt(i, e, (a) => {
    var s, n, l, d;
    const o = a.zattrs ?? {};
    return {
      shape: ((s = a.zarray) == null ? void 0 : s.shape) ?? [],
      chunks: ((n = a.zarray) == null ? void 0 : n.chunks) ?? ((l = a.zarray) == null ? void 0 : l.shape) ?? [],
      dtype: kt(((d = a.zarray) == null ? void 0 : d.dtype) ?? "|u1"),
      dims: o._ARRAY_DIMENSIONS ?? [],
      attrs: wt(o)
    };
  });
}
function $t(r) {
  const t = r.metadata ?? {}, e = /* @__PURE__ */ new Set();
  for (const o of Object.keys(t))
    o === ".zgroup" || o === ".zattrs" || o.endsWith("/.zgroup") && e.add(o.slice(0, -8));
  if (e.size === 0) {
    const o = B(t, "");
    if (!T(o))
      throw new Error("No arrays found in .zmetadata");
    return { groups: null, ...o };
  }
  const i = {}, a = B(t, "");
  T(a) && (i["/"] = a);
  for (const o of [...e].sort()) {
    const s = B(t, `${o}/`);
    T(s) && (i[o] = s);
  }
  if (!Object.keys(i).length)
    throw new Error("No arrays found in .zmetadata");
  return { groups: i };
}
function N(r, t) {
  const i = (r[t || ""] ?? {}).attributes ?? {}, a = {};
  for (const [o, s] of Object.entries(r))
    if (s.node_type === "array")
      if (t) {
        if (!o.startsWith(`${t}/`)) continue;
        const n = o.slice(t.length + 1);
        if (n.includes("/")) continue;
        a[n] = { zarray: s };
      } else {
        if (!o || o.includes("/")) continue;
        a[o] = { zarray: s };
      }
  return nt(a, i, (o) => {
    var n, l;
    const s = o.zarray;
    return {
      shape: s.shape ?? [],
      chunks: ((l = (n = s.chunk_grid) == null ? void 0 : n.configuration) == null ? void 0 : l.chunk_shape) ?? s.shape ?? [],
      // v3 dtypes are already human-readable.
      dtype: s.data_type ?? "float32",
      dims: s.dimension_names ?? [],
      attrs: s.attributes ?? {}
    };
  });
}
function zt(r) {
  var o;
  const t = ((o = r.consolidated_metadata) == null ? void 0 : o.metadata) ?? {};
  if (!Object.keys(t).length)
    throw new Error("zarr.json has no consolidated_metadata");
  const e = /* @__PURE__ */ new Set();
  for (const [s, n] of Object.entries(t))
    !s || n.node_type !== "group" || e.add(s);
  if (e.size === 0) {
    const s = N(t, "");
    if (!T(s))
      throw new Error("No arrays found in zarr.json");
    return { groups: null, ...s };
  }
  const i = {}, a = N(t, "");
  T(a) && (i["/"] = a);
  for (const s of [...e].sort()) {
    const n = N(t, s);
    T(n) && (i[s] = n);
  }
  if (!Object.keys(i).length)
    throw new Error("No arrays found in zarr.json");
  return { groups: i };
}
async function At(r, t = {}) {
  const e = st(r).replace(/\/$/, ""), a = { credentials: "same-origin", headers: (t.getAuthHeaders ?? G)() };
  let o = null, s = 0;
  try {
    const n = await fetch(`${e}/.zmetadata`, a);
    n.ok && (o = await n.json(), s = 2);
  } catch {
  }
  if (!o)
    try {
      const n = await fetch(`${e}/zarr.json`, a);
      n.ok && (o = await n.json(), s = 3);
    } catch {
    }
  if (!o)
    throw new Error("Could not read zarr metadata (.zmetadata or zarr.json)");
  return s === 2 ? $t(o) : zt(o);
}
function z(r) {
  return String(r).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
let Ct = 0;
function U() {
  return `xr${++Ct}`;
}
function tt(r) {
  return `<svg class="icon xr-${r}"><use xlink:href="#${r}"></use></svg>`;
}
function Mt(r, t) {
  return Object.keys(r).length ? `<ul class='xr-dim-list'>${Object.entries(r).map(
    ([i, a]) => `<li><span${t.has(i) ? " class='xr-has-index'" : ""}>${z(i)}</span>: ${a}</li>`
  ).join("")}</ul>` : "";
}
function dt(r) {
  const t = Object.entries(r);
  return t.length ? `<dl class='xr-attrs'>${t.map(([e, i]) => `<dt><span>${z(e)} :</span></dt><dd>${z(String(i))}</dd>`).join("")}</dl>` : "<em>No attributes</em>";
}
function Et(r) {
  const t = r.shape.reduce((e, i) => e * i, 1);
  return t === 0 ? "[]" : `${r.dtype} (${r.shape.join(" × ")} = ${t.toLocaleString()})`;
}
function et(r) {
  return r < 1024 ? r + " B" : r < 1024 ** 2 ? (r / 1024).toFixed(2) + " KiB" : r < 1024 ** 3 ? (r / 1024 ** 2).toFixed(2) + " MiB" : (r / 1024 ** 3).toFixed(2) + " GiB";
}
function Tt(r) {
  if (!r.length) return "";
  const t = Math.min(r.length, 3), e = r.slice(-t), i = (w) => Math.max(20, Math.min(110, 20 + Math.log10(Math.max(1, w)) * 30)), a = (w) => w.toLocaleString(), s = "font-size:11px;fill:var(--xr-font-color2);font-family:monospace";
  if (t < 3) {
    const w = i(e[t - 1]), C = t === 2 ? i(e[0]) : 12, M = t === 2 ? 42 : 2, S = 16, j = M + w + 4, O = C + S;
    return `<svg width="${Math.ceil(j)}" height="${Math.ceil(O)}"
        viewBox="0 0 ${Math.ceil(j)} ${Math.ceil(O)}"
        style="overflow:visible;display:block;flex-shrink:0">
      <rect x="${M}" y="0" width="${w}" height="${C}"
            style="fill:var(--xr-chunk-face);stroke:var(--xr-chunk-edge);stroke-width:0.8"/>
      ${t === 2 ? `<text x="${M - 5}" y="${C / 2 + 4}"
            text-anchor="end" style="${s}">${a(e[0])}</text>` : ""}
      <text x="${M + w / 2}" y="${C + S - 3}"
            text-anchor="middle" style="${s}">${a(e[t - 1])}</text>
    </svg>`;
  }
  const n = i(e[2]), l = i(e[1]), d = i(e[0]), h = 0.5, p = d * h, u = d * h * 0.45, y = 16, _ = 16, c = n + p + 52, k = y + l + u + _, g = 2, f = y + l + u;
  return `<svg width="${Math.ceil(c)}" height="${Math.ceil(k)}"
      viewBox="0 0 ${Math.ceil(c)} ${Math.ceil(k)}"
      style="overflow:visible;display:block;flex-shrink:0">
    <polygon points="${g},${f} ${g + n},${f} ${g + n},${f - l} ${g},${f - l}"
             style="fill:var(--xr-chunk-face);stroke:var(--xr-chunk-edge);stroke-width:0.8"/>
    <polygon points="${g},${f - l} ${g + n},${f - l} ${g + n + p},${f - l - u} ${g + p},${f - l - u}"
             style="fill:var(--xr-chunk-top);stroke:var(--xr-chunk-edge);stroke-width:0.8"/>
    <polygon points="${g + n},${f} ${g + n + p},${f - u} ${g + n + p},${f - l - u} ${g + n},${f - l}"
             style="fill:var(--xr-chunk-side);stroke:var(--xr-chunk-edge);stroke-width:0.8"/>
    <text x="${g + n / 2}" y="${f + _ - 3}"
          text-anchor="middle" style="${s}">${a(e[2])}</text>
    <text x="${g + n / 2 + p / 2}" y="${y - 3}"
          text-anchor="middle" style="${s}">${a(e[1])}</text>
    <text x="${g + n + p + 5}" y="${f - l / 2 - u / 2 + 4}"
          text-anchor="start" style="${s}">${a(e[0])}</text>
  </svg>`;
}
function St(r) {
  const { shape: t, chunks: e, dtype: i } = r, a = {
    int8: 1,
    uint8: 1,
    bool: 1,
    int16: 2,
    uint16: 2,
    int32: 4,
    uint32: 4,
    float32: 4,
    int64: 8,
    uint64: 8,
    float64: 8
  }[i] ?? 4, o = t.reduce((u, y) => u * y, 1) * a;
  let s = null, n = null;
  e && e.length === t.length && (s = e.reduce((u, y) => u * y, 1) * a, n = t.reduce((u, y, _) => u * Math.ceil(y / e[_]), 1));
  const l = 'style="color:var(--xr-font-color3);padding:2px 16px 2px 0;white-space:nowrap;vertical-align:top"', d = 'style="padding:2px 16px 2px 0;white-space:nowrap;vertical-align:top"', h = 'style="padding:2px 0;white-space:nowrap;color:var(--xr-font-color2);vertical-align:top"', p = (u, y, _ = "") => `<tr><td ${l}>${u}</td><td ${d}>${y}</td><td ${h}>${_}</td></tr>`;
  return `<table style="border-collapse:collapse;padding:6px 0 12px"><tr>
    <td style="vertical-align:top;padding:0">
      <table style="font-size:12px;font-family:monospace;border-collapse:collapse;line-height:1.75">
        <thead><tr>
          <th style="font-weight:400;padding:0 16px 5px 0;text-align:left"></th>
          <th style="font-weight:600;padding:0 16px 5px 0;text-align:left">Array</th>
          <th style="font-weight:600;padding:0 0 5px 0;text-align:left">Chunk</th>
        </tr></thead><tbody>
          ${p("Bytes", et(o), s !== null ? et(s) : "—")}
          ${p("Shape", "(" + t.join(", ") + ")", e ? "(" + e.join(", ") + ")" : "—")}
          ${n !== null ? p("Chunks", n.toLocaleString() + " chunks") : ""}
          ${p("dtype", z(i))}
          ${p("dims", "(" + r.dims.join(", ") + ")")}
        </tbody>
      </table>
    </td>
    <td style="vertical-align:middle;padding:0 0 0 32px">${Tt(t)}</td>
  </tr></table>`;
}
function jt(r, t, e) {
  const i = U(), a = U(), o = Object.keys(t.attrs).length > 0;
  return `
    <div class='xr-var-name'><span${e ? " class='xr-has-index'" : ""}>${z(r)}</span></div>
    <div class='xr-var-dims'>(${t.dims.map(z).join(", ")})</div>
    <div class='xr-var-dtype'>${z(t.dtype)}</div>
    <div class='xr-var-preview xr-preview'>${z(Et(t))}</div>
    <input id='${i}' class='xr-var-attrs-in' type='checkbox'${o ? "" : " disabled"}>
    <label for='${i}' title='Show/Hide attributes'>${tt("icon-file-text2")}</label>
    <input id='${a}' class='xr-var-data-in' type='checkbox'>
    <label for='${a}' title='Show/Hide data repr'>${tt("icon-database")}</label>
    <div class='xr-var-attrs'>${dt(t.attrs)}</div>
    <div class='xr-var-data'>${St(t)}</div>
  `;
}
function it(r, t) {
  return `<ul class='xr-var-list'>${Object.entries(r).map(
    ([e, i]) => `<li class='xr-var-item'>${jt(e, i, t.has(e))}</li>`
  ).join("")}</ul>`;
}
function R(r, t, e, i, a, o) {
  const s = U(), n = (i ?? 0) > 0, l = i !== null ? ` <span>(${i})</span>` : "", d = a && n ? "" : " disabled";
  return `
    <input id='${s}' class='xr-section-summary-in' type='checkbox'${d}${o || !n ? "" : " checked"} />
    <label for='${s}' class='xr-section-summary'${d === "" ? " title='Expand/collapse section'" : ""}>${r}${l}</label>
    <div class='xr-section-inline-details'>${t}</div>
    ${e ? `<div class='xr-section-details'>${e}</div>` : ""}
  `;
}
const rt = `<svg style="position:absolute;width:0;height:0;overflow:hidden"><defs>
<symbol id="icon-database" viewBox="0 0 32 32">
  <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"/>
  <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"/>
  <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"/>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
  <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"/>
</symbol>
</defs></svg>`;
function at(r) {
  const t = new Set(Object.keys(r.coords)), e = [];
  return e.push(
    R(
      "Dimensions:",
      Mt(r.dims, t),
      "",
      Object.keys(r.dims).length,
      !1,
      !0
    )
  ), Object.keys(r.coords).length && e.push(
    R(
      "Coordinates:",
      "",
      it(r.coords, t),
      Object.keys(r.coords).length,
      !0,
      !1
    )
  ), e.push(
    R(
      "Data variables:",
      "",
      it(r.data_vars, /* @__PURE__ */ new Set()),
      Object.keys(r.data_vars).length,
      !0,
      !1
    )
  ), Object.keys(r.attrs).length && e.push(
    R(
      "Attributes:",
      "",
      dt(r.attrs),
      Object.keys(r.attrs).length,
      !0,
      !0
    )
  ), `<div class='xr-root'>
    <div class='xr-wrap'>
      <div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div>
      <ul class='xr-sections'>${e.map((a) => `<li class='xr-section-item'>${a}</li>`).join("")}</ul>
    </div>
  </div>`;
}
function Ot(r) {
  if (!r.groups)
    return `${rt}${at(r)}`;
  const t = Object.entries(r.groups).map(
    ([e, i]) => `
    <details open style="margin-bottom:10px;border:1px solid var(--xr-border-color);border-radius:4px;overflow:hidden">
      <summary style="padding:8px 12px;font-weight:600;cursor:pointer;background:var(--xr-background-color-row-odd);list-style:none;display:flex;align-items:center;gap:8px">
        <span style="font-size:11px;color:var(--xr-font-color2)">▶</span>
        <span>Group: ${z(e)}</span>
      </summary>
      <div style="padding:0 12px 8px">${at(i)}</div>
    </details>
  `
  ).join("");
  return `${rt}<div style="font-family:monospace">${t}</div>`;
}
const It = `
:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0,0,0,1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0,0,0,.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0,0,0,.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
.xr-wrap{display:block!important;min-width:300px;max-width:700px;line-height:1.6;padding-bottom:4px}
.xr-header{padding-top:6px;padding-bottom:6px;border-bottom:solid 1px var(--xr-border-color);margin-bottom:4px}
.xr-header>div,.xr-header>ul{display:inline;margin-top:0;margin-bottom:0}
.xr-obj-type,.xr-obj-name{margin-left:2px;margin-right:10px}
.xr-obj-type{color:var(--xr-font-color2)}
.xr-sections{padding-left:0!important;display:grid;grid-template-columns:150px auto auto 1fr 0 20px 0 20px;margin-block-start:0;margin-block-end:0}
.xr-section-item{display:contents}
.xr-section-item>input,.xr-var-item>input{display:block;opacity:0;height:0;margin:0}
.xr-section-item>input+label,.xr-var-item>input+label{color:var(--xr-disabled-color)}
.xr-section-item>input:enabled+label,.xr-var-item>input:enabled+label{cursor:pointer;color:var(--xr-font-color2)}
.xr-section-item>input:enabled+label:hover,.xr-var-item>input:enabled+label:hover{color:var(--xr-font-color0)}
.xr-section-summary{grid-column:1;color:var(--xr-font-color2);font-weight:500;white-space:nowrap}
.xr-section-summary>span{display:inline-block;padding-left:.3em}
.xr-section-summary-in:disabled+label{color:var(--xr-font-color2)}
.xr-section-summary-in+label:before{display:inline-block;content:"►";font-size:11px;width:15px;text-align:center}
.xr-section-summary-in:disabled+label:before{color:var(--xr-disabled-color)}
.xr-section-summary-in:checked+label:before{content:"▼"}
.xr-section-summary-in:checked+label>span{display:none}
.xr-section-summary,.xr-section-inline-details{padding-top:4px}
.xr-section-inline-details{grid-column:2/-1}
.xr-section-details{grid-column:1/-1;margin-top:4px;margin-bottom:5px}
.xr-section-summary-in~.xr-section-details{display:none}
.xr-section-summary-in:checked~.xr-section-details{display:contents}
.xr-array-wrap{grid-column:1/-1;display:grid;grid-template-columns:20px auto}
.xr-array-wrap>label{grid-column:1;vertical-align:top}
.xr-preview{color:var(--xr-font-color3)}
.xr-array-preview,.xr-array-data{padding:0 5px!important;grid-column:2}
.xr-array-data,.xr-array-in:checked~.xr-array-preview{display:none}
.xr-array-in:checked~.xr-array-data,.xr-array-preview{display:inline-block}
.xr-dim-list{display:inline-block!important;list-style:none;padding:0!important;margin:0}
.xr-dim-list li{display:inline-block;padding:0;margin:0}
.xr-dim-list:before{content:"("}
.xr-dim-list:after{content:")"}
.xr-dim-list li:not(:last-child):after{content:",";padding-right:5px}
.xr-has-index{font-weight:bold}
.xr-var-list,.xr-var-item{display:contents}
.xr-var-item>div,.xr-var-item label,.xr-var-item>.xr-var-name span{background-color:var(--xr-background-color-row-even);border-color:var(--xr-background-color-row-odd);margin-bottom:0;padding-top:2px}
.xr-var-list>li:nth-child(odd)>div,.xr-var-list>li:nth-child(odd)>label,.xr-var-list>li:nth-child(odd)>.xr-var-name span{background-color:var(--xr-background-color-row-odd);border-color:var(--xr-background-color-row-even)}
.xr-var-name{grid-column:1}
.xr-var-dims{grid-column:2}
.xr-var-dtype{grid-column:3;text-align:right;color:var(--xr-font-color2)}
.xr-var-preview{grid-column:4}
.xr-index-preview{grid-column:2/5;color:var(--xr-font-color2)}
.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-right:10px}
.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover{overflow:visible;width:auto;z-index:1}
.xr-var-attrs,.xr-var-data,.xr-index-data{display:none;border-top:2px dotted var(--xr-background-color);padding-bottom:20px!important;padding-top:10px!important}
.xr-var-attrs-in:checked~.xr-var-attrs,.xr-var-data-in:checked~.xr-var-data{display:block}
.xr-var-data>table{float:right}
.xr-var-data>pre,.xr-var-data>table>tbody>tr{background-color:transparent!important}
.xr-var-name span,.xr-var-data,.xr-attrs{padding-left:25px!important}
.xr-attrs,.xr-var-attrs,.xr-var-data,.xr-index-data{grid-column:1/-1}
dl.xr-attrs{padding:0;margin:0;display:grid;grid-template-columns:125px auto}
.xr-attrs dt,.xr-attrs dd{padding:0;margin:0;float:left;padding-right:10px;width:auto}
.xr-attrs dt{font-weight:normal;grid-column:1}
.xr-attrs dd{grid-column:2;white-space:pre-wrap;word-break:break-all}
.xr-icon-database,.xr-icon-file-text2{display:inline-block;vertical-align:middle;width:1em;height:1.5em!important;stroke-width:0;stroke:currentColor;fill:currentColor}
.xr-var-attrs-in:checked+label>.xr-icon-file-text2,.xr-var-data-in:checked+label>.xr-icon-database{color:var(--xr-font-color0);filter:drop-shadow(1px 1px 5px var(--xr-font-color2));stroke-width:.8px}
.xr-var-item>input+label{cursor:pointer;color:var(--xr-font-color2);padding:0 1px}
`;
let ot = !1;
function Lt(r = {}) {
  if (ot || typeof document > "u") return;
  ot = !0;
  const t = typeof window < "u" ? window.MAIN_COLOR : void 0, e = (r.mainColor ?? t ?? "#9b7a52").replace(/^#/, ""), i = parseInt(e.slice(0, 2), 16), a = parseInt(e.slice(2, 4), 16), o = parseInt(e.slice(4, 6), 16), s = (p) => `rgb(${Math.min(255, i * p | 0)},${Math.min(255, a * p | 0)},${Math.min(255, o * p | 0)})`, n = (p, u) => `rgba(${Math.min(255, i * p | 0)},${Math.min(255, a * p | 0)},${Math.min(255, o * p | 0)},${u})`, l = `rgb(${255 - i},${255 - a},${255 - o})`, d = `
    :root{--xr-chunk-face:${s(0.85)};--xr-chunk-top:${s(1.25)};--xr-chunk-side:${s(0.55)};--xr-chunk-edge:${l}}
    html[data-theme="dark"],body[data-theme="dark"],body.vscode-dark{
      --xr-chunk-face:${n(0.85, 0.65)};--xr-chunk-top:${n(1.25, 0.65)};--xr-chunk-side:${n(0.55, 0.65)};--xr-chunk-edge:${l}
    }
  `, h = document.createElement("style");
  h.setAttribute("data-xarray-repr", "1"), h.textContent = d + It, document.head.appendChild(h);
}
async function Rt(r, t = {}) {
  t.injectCss !== !1 && Lt({ mainColor: t.mainColor });
  const e = await At(r, { getAuthHeaders: t.getAuthHeaders });
  return Ot(e);
}
export {
  ft as AggregationConfigElement,
  mt as DataInspectorElement,
  E as NcDumpDialogState,
  xt as ZarrLoadingStepsElement,
  qt as ZarrPoller,
  Ot as buildXarrayRepr,
  Dt as detectZarrStore,
  Lt as injectXarrayCss,
  Rt as loadZarrMetadataHtml,
  At as openDatasetMeta
};
