/* Waterpark dataset tree */
(() => {
    "use strict";
  
    const SVG = {
      chevron: '<svg class="wp__chev" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M6 4l4 4-4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
      dir:     '<svg class="wp__icon wp__icon--dir" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M1.75 3.5A1.25 1.25 0 013 2.25h3l1.5 1.5h4.5A1.25 1.25 0 0113.25 5v6.75A1.25 1.25 0 0112 13H3a1.25 1.25 0 01-1.25-1.25V3.5z"/></svg>',
      store:   '<svg class="wp__icon wp__icon--store" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 1.2l6 3v7.6l-6 3-6-3V4.2l6-3zM8 2.9L3.6 5.1 8 7.3l4.4-2.2L8 2.9zM3 6.3v5l4.4 2.2V8.5L3 6.3zm5.6 7.2L13 11.3v-5L8.6 8.5v5z"/></svg>',
      link:    '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M6.5 9.5l3-3M7 4h3a2 2 0 012 2v0M9 12H6a2 2 0 01-2-2v0" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>',
      search:  '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="7" cy="7" r="4.5" stroke="currentColor" stroke-width="1.5"/><path d="M10.5 10.5L14 14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    };
    const BADGE = { available: ["wp__badge--ok", "available"], "in-progress": ["wp__badge--wip", "in progress"], planned: ["wp__badge--wip", "planned"] };
  
    const humanBytes = (n) => {
      if (n == null || isNaN(n)) return "";
      const u = ["B", "KB", "MB", "GB", "TB", "PB"]; let i = 0; n = Number(n);
      while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
      return (n >= 100 || i === 0 ? Math.round(n) : n.toFixed(1)) + " " + u[i];
    };
    const h = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  
    function copy(text, btn) {
      const done = () => { btn.classList.add("is-done"); const t = btn.textContent; btn.textContent = "Copied"; setTimeout(() => { btn.classList.remove("is-done"); btn.textContent = t; }, 1200); };
      if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).then(done, done);
      else { const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); done(); }
    }
  
    /*listing back-ends: each returns {dirs:[node], stores:[node]} */
    const xmlText = (el, tag) => { const n = el.getElementsByTagName(tag)[0]; return n ? n.textContent : null; };
  
    async function s3ListBuckets(endpoint) {
      const res = await fetch(endpoint + "/");
      if (!res.ok) throw new Error("ListBuckets HTTP " + res.status);
      const xml = new DOMParser().parseFromString(await res.text(), "application/xml");
      return [...xml.getElementsByTagName("Bucket")].map((b) => xmlText(b, "Name")).filter(Boolean);
    }
  
    async function s3List(endpoint, bucket, prefix) {
      const dirs = [], stores = [];
      let token = null;
      do {
        const u = new URL(endpoint + "/" + bucket);
        u.searchParams.set("list-type", "2");
        u.searchParams.set("delimiter", "/");
        if (prefix) u.searchParams.set("prefix", prefix);
        if (token) u.searchParams.set("continuation-token", token);
        const res = await fetch(u.toString());
        if (!res.ok) throw new Error("HTTP " + res.status);
        const xml = new DOMParser().parseFromString(await res.text(), "application/xml");
        for (const cp of xml.getElementsByTagName("CommonPrefixes")) {
          const p = xmlText(cp, "Prefix");
          const name = p.replace(/\/$/, "").split("/").pop();
          const base = { name, bucket, prefix: p, path: bucket + "/" + p.replace(/\/$/, "") };
          if (/\.zarr$/i.test(name)) stores.push({ ...base, type: "store", meta: {} });
          else dirs.push({ ...base, type: "dir" });
        }
        for (const c of xml.getElementsByTagName("Contents")) {
          const key = xmlText(c, "Key");
          if (!key || key === prefix || /\/$/.test(key)) continue;
          stores.push({ type: "store", name: key.split("/").pop(), path: bucket + "/" + key, size: +(xmlText(c, "Size") || 0), meta: {} });
        }
        const trunc = xmlText(xml, "IsTruncated") === "true";
        token = trunc ? xmlText(xml, "NextContinuationToken") : null;
      } while (token);
      return { dirs, stores };
    }
  
    async function apiList(api, path) {
      const u = new URL(api, location.href);
      if (path) u.searchParams.set("path", path);
      const res = await fetch(u.toString());
      if (!res.ok) throw new Error("HTTP " + res.status);
      const j = await res.json();
      return { dirs: (j.dirs || []).map((d) => ({ ...d, type: "dir" })), stores: (j.stores || []).map((s) => ({ ...s, type: "store", meta: s.meta || {} })) };
    }
  
    function listChildren(ctx, node) {
      if (ctx.mode === "api") return apiList(ctx.api, node.type === "bucket" ? node.bucket : node.path);
      return s3List(ctx.endpoint, node.bucket, node.prefix || "");
    }
  
    // node rendering
    function nodeEl(node, ctx) {
      const li = h("li");
      const isStore = node.type === "store";
      const isBucket = node.type === "bucket";
      const expandable = !isStore;
  
      const row = h("button", "wp__row"); row.type = "button";
      row.innerHTML = SVG.chevron;
      if (!expandable) row.querySelector(".wp__chev").classList.add("wp__chev--leaf");
      row.insertAdjacentHTML("beforeend", isStore ? SVG.store : SVG.dir);
  
      const name = h("span", "wp__name" + (isStore ? " wp__name--store" : ""));
      if (isBucket) {
        name.classList.add("wp__bucket-title");
        name.textContent = node.title || node.name;
        if (node.label) { const s = h("span", "wp__bucket-sub"); s.textContent = "  " + node.label; name.appendChild(s); }
      } else name.textContent = node.name;
      row.appendChild(name);
  
      if (isBucket && node.href) {
        const a = h("a", "wp__doclink", SVG.link);
        a.href = node.href; a.target = "_blank"; a.rel = "noopener"; a.title = "Project page";
        a.addEventListener("click", (e) => e.stopPropagation());
        row.appendChild(a);
      }
      row.appendChild(h("span", "wp__spacer"));
      if (isBucket && node.status && BADGE[node.status]) { const [c, t] = BADGE[node.status]; const b = h("span", "wp__badge " + c); b.textContent = t; row.appendChild(b); }
      if (isStore) {
        const lvl = node.meta && node.meta.healpix_level;
        if (lvl != null) { const l = h("span", "wp__lvl"); l.textContent = "HP " + lvl; row.appendChild(l); }
        if (node.size != null) { const s = h("span", "wp__size"); s.textContent = humanBytes(node.size); row.appendChild(s); }
      }
      li.appendChild(row);
      li._node = node; li._row = row; li._name = (node.title || node.name || "").toLowerCase();
  
      if (isStore) {
        const detail = buildDetail(node, ctx); detail.hidden = true; li.appendChild(detail);
        row.setAttribute("aria-expanded", "false");
        row.addEventListener("click", () => { const open = detail.hidden; detail.hidden = !open; row.setAttribute("aria-expanded", String(open)); });
        return li;
      }
  
      const kids = h("ul", "wp__children"); kids.hidden = true; li.appendChild(kids);
      row.setAttribute("aria-expanded", "false");
      li._kids = kids; li._loaded = false;
      row.addEventListener("click", async () => {
        const open = kids.hidden;
        if (open && !li._loaded) { li._loaded = true; await populate(li, node, ctx); }
        kids.hidden = !open; row.setAttribute("aria-expanded", String(open));
      });
      return li;
    }
  
    async function populate(li, node, ctx) {
      const kids = li._kids;
      if (!ctx.lazy) { (node.children || []).forEach((c) => kids.appendChild(nodeEl(c, ctx))); return; }
      const wait = h("div", "wp__loading", '<span class="wp__spin"></span> listing…'); kids.appendChild(wait);
      try {
        const { dirs, stores } = await listChildren(ctx, node);
        wait.remove();
        dirs.sort((a, b) => a.name.localeCompare(b.name)).forEach((d) => kids.appendChild(nodeEl(d, ctx)));
        stores.sort((a, b) => a.name.localeCompare(b.name)).forEach((s) => kids.appendChild(nodeEl(s, ctx)));
        if (!kids.children.length) kids.appendChild(h("div", "wp__empty", "empty"));
      } catch (err) {
        wait.remove();
        kids.appendChild(h("div", "wp__error", "Could not list — " + err.message + (ctx.mode === "live" ? " (likely CORS or anonymous listing is disabled on the endpoint)" : "")));
      }
    }
  
    function buildDetail(node, ctx) {
      const wrap = h("div", "wp__detail");
      const s3path = "s3://" + node.path;
      const httpUrl = ctx && ctx.endpoint ? ctx.endpoint.replace(/\/$/, "") + "/" + node.path : null;
  
      // An effort to keep actions live on their own row so they never scroll out of view, with
      // Inspect as the primary action
      const actions = h("div", "wp__actions");
      const fname = h("span", "wp__fname"); fname.textContent = node.name;
      const cb = h("button", "wp__copy"); cb.type = "button"; cb.textContent = "Copy path";
      cb.addEventListener("click", () => copy(s3path, cb));
      actions.append(fname, h("span", "wp__spacer"), cb);
      if (httpUrl && typeof window.WATERPARK_INSPECT === "function") {
        const ib = h("button", "wp__copy wp__inspect"); ib.type = "button"; ib.textContent = "Inspect";
        ib.addEventListener("click", () => window.WATERPARK_INSPECT({ url: httpUrl, name: node.name, path: node.path }));
        actions.append(ib);
      }
      wrap.appendChild(actions);
  
      // Full-width path pill that scrolls horizontally on its own
      const path = h("div", "wp__path", `<code>${s3path}</code>`);
      wrap.appendChild(path);
  
      const m = node.meta || {}; const meta = h("div", "wp__meta");
      if (m.dims && Object.keys(m.dims).length) {
        const r = h("div", "wp__metarow", '<span class="wp__metakey">dims</span>');
        Object.entries(m.dims).forEach(([k, v]) => r.appendChild(h("span", "wp__chip", `${k} <b>${Number(v).toLocaleString()}</b>`)));
        meta.appendChild(r);
      }
      if (m.vars && m.vars.length) {
        const r = h("div", "wp__metarow", '<span class="wp__metakey">vars</span>');
        m.vars.slice(0, 12).forEach((v) => { const c = h("span", "wp__chip"); c.textContent = v; r.appendChild(c); });
        if (m.vars.length > 12) { const c = h("span", "wp__chip"); c.textContent = "+" + (m.vars.length - 12); r.appendChild(c); }
        meta.appendChild(r);
      }
      if (m.title) { const r = h("div", "wp__metarow", '<span class="wp__metakey">title</span>'); const c = h("span", "wp__bucket-sub"); c.textContent = m.title; r.appendChild(c); meta.appendChild(r); }
      if (meta.children.length) wrap.appendChild(meta);
      return wrap;
    }
  
    // filter (searches loaded levels; static also pulls children in)
    function filter(rootUl, q, ctx) {
      q = q.trim().toLowerCase();
      const walk = (ul) => {
        let any = false;
        [...ul.children].forEach((li) => {
          if (!li._row) return;
          const selfHit = q && li._name.includes(q);
          let kidHit = false;
          if (li._kids) {
            if (q && !li._loaded && !ctx.lazy && li._node) { li._loaded = true; (li._node.children || []).forEach((c) => li._kids.appendChild(nodeEl(c, ctx))); }
            kidHit = walk(li._kids);
            if (q) { li._kids.hidden = !kidHit; li._row.setAttribute("aria-expanded", String(kidHit)); }
            else { li._kids.hidden = true; li._row.setAttribute("aria-expanded", "false"); }
          }
          const show = !q || selfHit || kidHit;
          li.classList.toggle("wp__hidden", !show);
          li._row.classList.toggle("wp__hit", !!(q && selfHit));
          any = any || show;
        });
        return any;
      };
      walk(rootUl);
    }
  
    //mount
    async function loadData(host, ctx) {
      if (window.WATERPARK_DATA) return window.WATERPARK_DATA;
      if (ctx.mode === "static") {
        const res = await fetch(host.dataset.src || "assets/waterpark-index.json");
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      }
      // Shared metadata
      async function fetchMeta(url, timeoutMs) {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), timeoutMs || 2500);
        try {
          const r = await fetch(url, { signal: ctrl.signal });
          if (!r.ok) throw new Error("HTTP " + r.status);
          return await r.json();
        } finally {
          clearTimeout(t);
        }
      }
  
      let metaRaw = {};
      // first fetch raw GitHub, then automatically fall back to the 
      // same-named file shipped locally in assets/ then get the bucket
      // only names from md file
      const metaSources = [];
      if (host.dataset.meta) {
        metaSources.push(host.dataset.meta);
        const localName = host.dataset.meta.split("/").pop().split("?")[0];
        if (localName) metaSources.push("../assets/" + localName);
      }
      for (const url of metaSources) {
        try { metaRaw = await fetchMeta(url, 2500); break; }
        catch (_) { /* try the next source */ }
      }
      const meta = metaRaw.datasets || metaRaw || {};

      // lazy modes: build the bucket list, children load on expand
      let buckets = (Array.isArray(metaRaw.buckets) ? metaRaw.buckets : [])
        .map((s) => String(s).trim()).filter(Boolean);
      if (!buckets.length) {
        buckets = (host.dataset.buckets || "").split(",").map((s) => s.trim()).filter(Boolean);
      }
      if (!buckets.length) {
        if (ctx.mode === "live") buckets = await s3ListBuckets(ctx.endpoint);
        else {
          const top = await apiList(ctx.api, "");
          return { endpoint: ctx.endpoint, datasets: top.dirs.map((d) => { const b = d.bucket || d.name; return { type: "bucket", bucket: b, name: b, title: b, path: b, prefix: "", status: "available", ...d, ...(meta[b] || {}) }; }) };
        }
      }
      return { endpoint: ctx.endpoint, datasets: buckets.map((b) => ({ type: "bucket", bucket: b, name: b, title: b, path: b, prefix: "", status: "available", ...(meta[b] || {}) })) };
    }
  
    async function mount(host) {
      const mode = host.dataset.mode || "static";
      const ctx = { mode, lazy: mode !== "static", endpoint: (host.dataset.endpoint || "").replace(/\/$/, ""), api: (host.dataset.api || "").replace(/\/$/, "") };
  
      const bar = h("div", "wp__bar");
      const search = h("div", "wp__search", SVG.search);
      const input = h("input"); input.type = "search"; input.placeholder = "Filter datasets and paths…"; input.setAttribute("aria-label", "Filter"); search.appendChild(input);
      const expand = h("button", "wp__btn"); expand.type = "button"; expand.textContent = "Expand all";
      if (ctx.lazy) expand.classList.add("wp__hidden");  // no bulk expand when children load on demand
      const collapse = h("button", "wp__btn"); collapse.type = "button"; collapse.textContent = "Collapse";
      const liveMode = mode === "live" || mode === "api";
      bar.append(search, expand, collapse); host.appendChild(bar);
  
      const treeWrap = h("div", "wp__tree"); const rootUl = h("ul", "wp__root"); rootUl._ctx = ctx; treeWrap.appendChild(rootUl);
      const foot = h("div", "wp__foot wp__foot--status");
      const loading = h("div", "wp__loading", '<span class="wp__spin"></span> loading datasets…');
      host.appendChild(loading);
  
      let data;
      try { data = await loadData(host, ctx); }
      catch (err) { loading.replaceWith(h("div", "wp__error", "Could not load datasets — " + err.message + (ctx.mode === "live" ? " (anonymous ListBuckets may be disabled; set data-buckets, or use static/api mode)" : ""))); return; }
  
      loading.replaceWith(treeWrap); host.appendChild(foot);
      if (!ctx.endpoint && data.endpoint) ctx.endpoint = data.endpoint.replace(/\/$/, "");
      (data.datasets || []).forEach((d) => {
        const node = Object.assign({ type: "bucket", name: d.bucket, path: d.bucket || d.name, prefix: "" }, d);
        if (!ctx.lazy && d.tree) node.children = d.tree.children || [];
        rootUl.appendChild(nodeEl(node, ctx));
      });
  
      // Footer
      const fbadge = h("span", "wp__mode " + (liveMode ? "wp__mode--live" : "wp__mode--snap"),
        '<span class="wp__dot"></span>' + (liveMode ? "LIVE" : "SNAPSHOT"));
      const fdesc = h("span", "wp__foot-desc");
      fdesc.innerHTML = liveMode
        ? `${ctx.mode === "api" ? "via backend" : "direct S3 listing"}${data.endpoint ? " of <code>" + data.endpoint + "</code>" : ""}`
        : `${data.generated ? new Date(data.generated).toISOString().slice(0, 16).replace("T", " ") + " UTC" : "prebuilt index"}${data.endpoint ? " · <code>" + data.endpoint + "</code>" : ""}`;
      foot.append(fbadge, fdesc);
  
      expand.addEventListener("click", () => { if (ctx.lazy) return; rootUl.querySelectorAll("li").forEach((li) => { if (li._kids && li._kids.hidden) li._row.click(); }); });
      collapse.addEventListener("click", () => {
        input.value = "";
        [...rootUl.children].forEach((li) => { if (li._kids) { li._kids.hidden = true; li._row.setAttribute("aria-expanded", "false"); } });
        rootUl.querySelectorAll(".wp__hidden").forEach((e) => e.classList.remove("wp__hidden"));
        rootUl.querySelectorAll(".wp__hit").forEach((e) => e.classList.remove("wp__hit"));
      });
      let t; input.addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => filter(rootUl, input.value, ctx), 120); });
    }
  
    function init() { document.querySelectorAll("[data-waterpark]").forEach((el) => { if (!el._wpInit) { el._wpInit = true; mount(el); } }); }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
    if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);
  })();
