/*
   Waterpark inspector add-on
   Module path defaults to a sibling file in cdn
*/
   (function () {
    "use strict";
    var CDN_PKG = "@freva-org/" + "data-inspector";
    var CDN_VER = "3.1.0";
    var MODULE_URL = window.WATERPARK_INSPECTOR_SRC ||
      ("https://cdn.jsdelivr.net/npm/" + CDN_PKG + "@" + CDN_VER + "/+esm");
  

    var HEADER_BG = window.WATERPARK_HEADER_BG || "var(--md-primary-fg-color)";
    var LOAD_BG = window.WATERPARK_LOAD_BG || "#0d6efd";
  
    injectThemeCss();
  
    var modPromise = null;
    function loadModule() { return (modPromise = modPromise || import(MODULE_URL)); }
  
    var el = null, Nc = null;
  
    async function ensureElement() {
      var mod = await loadModule();
      Nc = mod.NcDumpDialogState || { LOADING: "loading", READY: "ready", ERROR: "error" };
      if (!el) {
        el = document.createElement("data-inspector");
        document.body.appendChild(el);
        el.addEventListener("inspector-submit", function (e) {
          var file = e.detail && e.detail.file;
          if (file) run(Array.isArray(file) ? file[0] : file);
        });
        el.addEventListener("inspector-close", function () { el.open = false; });
      }
      return mod;
    }
  
    async function run(url) {
      var mod = await ensureElement();
      el.file = url;
      // enables 3D Viewer
      el.zarrUrl = url;
      el.output = null;
      el.error = null;
      el.status = Nc.LOADING;
      el.open = true;
      try {
        var html = await mod.loadZarrMetadataHtml(url, {
          getAuthHeaders: function () { return {}; },
        });
        el.output = html;
        el.status = Nc.READY;
      } catch (err) {
        el.error = String(err && err.message || err);
        el.status = Nc.ERROR;
      }
    }
  
    // Hook the tree's Inspect button calls.
    window.WATERPARK_INSPECT = function (detail) { return run(detail.url); };
    function injectThemeCss() {
      if (document.getElementById("wp-insp-theme")) return;
      var chrome =
        "--di-bg:var(--md-default-bg-color);" +
        "--di-fg:var(--md-typeset-color);" +
        "--di-muted:var(--md-default-fg-color--light);" +
        "--di-border:var(--md-default-fg-color--lightest);" +
        "--di-surface:color-mix(in srgb,var(--md-typeset-color) 5%,var(--md-default-bg-color));" +
        "--di-accent:var(--md-primary-fg-color);";
      var reprDark =
        "--xr-font-color0:var(--md-typeset-color);" +
        "--xr-font-color2:var(--md-typeset-color);" +
        "--xr-font-color3:var(--md-default-fg-color--light);" +
        "--xr-border-color:var(--md-default-fg-color--lightest);" +
        "--xr-disabled-color:var(--md-default-fg-color--light);" +
        "--xr-background-color:var(--md-default-bg-color);" +
        "--xr-background-color-row-even:var(--md-default-bg-color);" +
        "--xr-background-color-row-odd:color-mix(in srgb,var(--md-typeset-color) 8%,var(--md-default-bg-color));";
      var css =
        "data-inspector{" + chrome + "}" +
        "[data-md-color-scheme] data-inspector{" + chrome + "}" +
        '[data-md-color-scheme="slate"] data-inspector{' + reprDark + "}" +
        "data-inspector .di-header{background:" + HEADER_BG + ";border-bottom:none;}" +
        "data-inspector .di-title{color:#fff;}" +
        "data-inspector .di-title-ico{color:#fff;}" +
        "data-inspector .di-close{color:#fff;}" +
        "data-inspector .di-close:hover{background:rgba(255,255,255,.2);color:#fff;}" +
        "data-inspector .di-pathbar-label{color:#fff;}" +
        "data-inspector .di-zarr-row{background:rgba(255,255,255,.15);}" +
        "data-inspector .di-zarr-row .di-muted{color:#fff;}" +
        "data-inspector .di-btn-primary{background:" + LOAD_BG + ";border-color:" + LOAD_BG + ";color:#fff;}" +

        "data-inspector{" +
          "--xr-chunk-face:color-mix(in srgb,var(--md-primary-fg-color) 85%,#fff);" +
          "--xr-chunk-top:color-mix(in srgb,var(--md-primary-fg-color) 70%,#fff);" +
          "--xr-chunk-side:color-mix(in srgb,var(--md-primary-fg-color) 60%,#000);" +
          "--xr-chunk-edge:var(--md-primary-fg-color);" +
        "}" +
        // Full-width metadata and lift the repr's built-in 700px cap
        "data-inspector .xr-wrap{max-width:none!important;}";
      var s = document.createElement("style");
      s.id = "wp-insp-theme"; s.textContent = css;
      document.head.appendChild(s);
    }
  })();
