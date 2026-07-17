/*
 * Freva Data Browser host loader.
  * Globally registered via `extra_javascript`, but every action is gated on the
  * presence of #databrowser-mount
 */
(function () {
  "use strict";

  // Module-scoped so we survive Material's instant-nav re-runs.
  var handle = null;
  var mountedEl = null;
  var booting = false;
  var schemeObserver = null;

  // helpers
  function parseConfigModule(mod) {
    // Support `export default {...}` and a named `config` export.
    if (!mod) return {};
    if (mod.default && typeof mod.default === "object") return mod.default;
    if (mod.config && typeof mod.config === "object") return mod.config;
    return {};
  }

  function pickMount(mod) {
    // The package exposes both a named and a default `mountDataBrowser`.
    return (mod && mod.mountDataBrowser) || (mod && mod.default) || null;
  }

  // Dynamic import() can reject on a transient network hiccup (a reset chunk, a
  // flaky dev server). The widget is bundled into ONE file, so this is now a
  // single request
  function importWithRetry(url, attempts) {
    attempts = attempts || 3;
    return import(/* @vite-ignore */ url).catch(function (err) {
      if (attempts <= 1) throw err;
      return new Promise(function (resolve) {
        setTimeout(resolve, 250);
      }).then(function () {
        return importWithRetry(url, attempts - 1);
      });
    });
  }

  // Waterpark's Material palette: scheme "slate" = dark, "default" = light.
  function currentMode() {
    var scheme =
      document.body.getAttribute("data-md-color-scheme") ||
      document.documentElement.getAttribute("data-md-color-scheme") ||
      "default";
    return scheme === "slate" ? "night" : "day";
  }

  // Move the mount so it is NOT a descendant of `.md-typeset`, so Material's
  // typeset element rules (h1/kbd/pre/line-height) can't bleed into the widget's
  // px-based, low-specificity styling.
  function relocateOutOfTypeset(mount) {
    if (mount.getAttribute("data-relocated") === "1") return;
    var typeset = mount.closest(".md-typeset");
    if (!typeset) return;
    var host =
      typeset.closest(".md-content") ||
      document.querySelector(".md-main__inner") ||
      document.querySelector(".md-main") ||
      document.body;
    host.appendChild(mount);
    mount.setAttribute("data-relocated", "1");
  }

  function syncTheme() {
    if (handle) {
      try {
        handle.setTheme(currentMode());
      } catch (e) {
        /* noop */
      }
    }
  }

  function watchScheme() {
    if (schemeObserver || !window.MutationObserver) return;
    schemeObserver = new MutationObserver(syncTheme);
    // Material sets data-md-color-scheme on <body>
    schemeObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
    schemeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
  }

  function teardown() {
    if (schemeObserver) {
      schemeObserver.disconnect();
      schemeObserver = null;
    }
    if (handle) {
      try {
        handle.destroy();
      } catch (e) {
        /* noop — never let teardown break navigation */
      }
      handle = null;
    }
    mountedEl = null;
    document.body.classList.remove("waterpark-databrowser-page");
  }

  // boot
  async function boot() {
    var mount = document.getElementById("databrowser-mount");

    if (!mount) {
      teardown();
      return;
    }

    // Already mounted into this exact node
    if (handle && mountedEl === mount) return;

    // The node changed
    if (handle && mountedEl !== mount) teardown();

    if (booting) return;
    booting = true;

    document.body.classList.add("waterpark-databrowser-page");

    // Escape Material's typeset wrapper BEFORE mounting
    relocateOutOfTypeset(mount);

    var entry = mount.getAttribute("data-databrowser-entry");
    var configUrl = mount.getAttribute("data-databrowser-config");
    var inspectorUrl = mount.getAttribute("data-databrowser-inspector");
    if (!entry) {
      booting = false;
      return;
    }

    try {
      var results = await Promise.all([
        importWithRetry(entry, 3),
        configUrl
          ? importWithRetry(configUrl, 3).catch(function () {
              return {};
            })
          : Promise.resolve({}),
      ]);

      // important re-check: the user may have navigated away while we were importing.
      var stillHere = document.getElementById("databrowser-mount");
      if (!stillHere || stillHere !== mount) {
        booting = false;
        boot();
        return;
      }

      var mountFn = pickMount(results[0]);
      if (typeof mountFn !== "function") {
        throw new Error("mountDataBrowser export not found in " + entry);
      }

      // Merge host-provided runtime bits onto the static config
      var config = parseConfigModule(results[1]);
      // Self-hosted inspector
      if (inspectorUrl) config.inspectorUrl = inspectorUrl;
      // Open in Waterpark's current light/dark mode from the first paint
      config.theme = config.theme || {};
      config.theme.mode = currentMode();

      handle = mountFn(mount, config);
      mountedEl = mount;

      // Keep following Waterpark's palette toggle for the rest of the visit.
      syncTheme();
      watchScheme();
    } catch (e) {
      // Surface a message instead of a blank page.
      if (mount && !mount.firstChild) {
        var msg = document.createElement("div");
        msg.className = "waterpark-databrowser-error";
        msg.textContent =
          "The Data Browser could not be loaded. Please try again later.";
        mount.appendChild(msg);
      }
      if (window.console && console.error) {
        console.error("[databrowser] mount failed:", e);
      }
    } finally {
      booting = false;
    }
  }

  // Material publishes `document$` on every navigation
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(boot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
