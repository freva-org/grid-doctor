/*
 * STAC Browser host loader
 * Globally registered via `extra_javascript`, but every action is gated on the
 * presence of #stac-browser-mount
 */
(function () {
  "use strict";
  // Promise of the first dynamic import()
  var entryLoaded = null;

  function ensureMetaTags() {
    var metas = [
      ["meta-description", { name: "description", content: "" }],
      ["og-title", { property: "og:title", content: document.title || "STAC Browser" }],
      ["og-description", { property: "og:description", content: "" }],
      ["og-locale", { property: "og:locale", content: "en" }],
      ["og-url", { property: "og:url", content: window.location.href }],
    ];
    for (var i = 0; i < metas.length; i++) {
      var id = metas[i][0];
      var attrs = metas[i][1];
      var el = document.getElementById(id);
      if (!el) {
        el = document.createElement("meta");
        el.id = id;
        document.head.appendChild(el);
      }
      for (var key in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, key)) {
          el.setAttribute(key, attrs[key]);
        }
      }
    }
  }

  function hasCss(href) {
    var links = document.querySelectorAll("link[data-stac-css]");
    for (var i = 0; i < links.length; i++) {
      if (links[i].getAttribute("data-stac-css") === href) return true;
    }
    return false;
  }

  function ensureCss(hrefs) {
    (hrefs || []).forEach(function (href) {
      if (!href) return;
      if (hasCss(href)) return;
      var link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = href;
      link.setAttribute("data-stac-css", href);
      document.head.appendChild(link);
    });
  }

  function parseList(value) {
    if (!value) return [];
    try {
      var parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : [parsed];
    } catch (e) {
      return [value];
    }
  }

  // Toggle a body class while STAC is showing its ROOT/landing catalog (hash
  // path "/"), so host CSS can hide the duplicate page title there but keep it
  // on collection pages
  function markRoute() {
    var h = (window.location.hash || "").replace(/^#/, "");
    var path = h.split("?")[0].replace(/\/+$/, "");
    var isRoot = path === "" || path === "/";
    document.body.classList.toggle("waterpark-stac-root", isRoot);
  }

  function teardown() {
    window.removeEventListener("hashchange", markRoute);
    document.body.classList.remove("waterpark-stac-page");
    document.body.classList.remove("waterpark-stac-root");
    // left behind if Browse was open
    document.body.classList.remove("stac-browser-sidebar");
    if (window.__WATERPARK_STAC_APP__) {
      try { window.__WATERPARK_STAC_APP__.unmount(); } catch (e) { /* noop */ }
      window.__WATERPARK_STAC_APP__ = null;
    }
  }

  async function boot() {
    var mount = document.getElementById("stac-browser-mount");

    // Not on the STAC page: clean up and stop.
    if (!mount) {
      teardown();
      return;
    }

    document.body.classList.add("waterpark-stac-page");
    markRoute();
    window.addEventListener("hashchange", markRoute);
    ensureMetaTags();
    ensureCss(parseList(mount.dataset.stacCss));

    // catalogUrl and other settings come from stac-browser.config.js
    var entry = mount.dataset.stacEntry;
    if (!entry) return;

    if (!entryLoaded) {
      entryLoaded = import(/* @vite-ignore */ entry).catch(function (e) {
        entryLoaded = null;
        throw e;
      });
      // IMPORTANT: main.js calls init() once on first evaluation
      await entryLoaded;
    } else {
      await entryLoaded;
      if (typeof window.STAC_BROWSER_INIT === "function") {
        // remount after an instant-nav return
        await window.STAC_BROWSER_INIT();
      }
    }
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(boot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
