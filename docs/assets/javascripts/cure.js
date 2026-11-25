/*
 * "The cure": an animation of a globe whose lat/lon grid heals into
 * HEALPix, for the landing page of the grid-doctor docs.
 *
 * Draws into every <canvas data-gd-cure> on the page, at the canvas's
 * display size times the device pixel ratio.  The globe spins a quarter
 * turn per 8-second loop (both grids repeat every 90 degrees, so the
 * loop is seamless):
 *
 *   0.0 - 2.0  the patient: lat/lon cells crowd at the poles, where
 *              "fever" cells (missing data) glow red
 *   2.0 - 4.6  the cure spreads, poles first and in a scattered order:
 *              lat/lon lines fade out, the 12 HEALPix base cells fade in
 *   4.0 - 6.0  every cell splits into 4, then 16 (HEALPix levels 1, 2)
 *   6.0 - 7.2  HEALed: a soft glow on some cells
 *   7.2 - 8.0  relapse: the lat/lon grid returns
 *
 * Cell edges are exact: they are traced from HEALPix face coordinates
 * (x, y) in [0, 1]^2 of the 12 base faces.  Colours follow the docs
 * theme (gold on navy in dark mode, violet on white in light mode).
 * With "reduce motion" a healed frame is shown as a still image.  The
 * animation pauses while it is off screen.
 */
(function () {
  "use strict";

  var LOOP = 8.0, TILT = 0.42, STILL = 6.6;
  var STEP = Math.PI / 12; // lat/lon grid spacing: 15 degrees
  var HALF_PI = Math.PI / 2;

  var PALETTES = {
    dark: {
      ink: [251, 191, 36], sick: [100, 116, 139], fever: [248, 113, 113],
      bodyIn: [30, 46, 72], bodyOut: [15, 27, 45], rim: [148, 163, 184],
    },
    light: {
      ink: [124, 58, 237], sick: [148, 163, 184], fever: [220, 38, 38],
      bodyIn: [255, 255, 255], bodyOut: [226, 232, 240], rim: [148, 163, 184],
    },
  };
  function schemeOf(canvas) {
    var el = canvas.closest("[data-md-color-scheme]") || document.body;
    var scheme = el.getAttribute("data-md-color-scheme");
    if (scheme === "slate") return "dark";
    if (scheme === "default") return "light";
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function phase(t, a, b) { return clamp01((t - a) / (b - a)); }
  function smooth(v) { v = clamp01(v); return v * v * (3 - 2 * v); }
  function hash(a, b) {
    var h = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
    return h - Math.floor(h);
  }
  function rgba(c, a) { return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a.toFixed(3) + ")"; }

  // ------------------------------------------------------------ geometry
  // Point on face f at face coordinates (x, y) in [0, 1]: [lat, lon].
  // The continuous form of HEALPix's pix2ang for nside = 1.
  var JRLL = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
  var JPLL = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];
  function faceToLatLon(f, x, y) {
    var jr = JRLL[f] - x - y, nr, z;
    if (jr < 1) { nr = jr; z = 1 - nr * nr / 3; }
    else if (jr > 3) { nr = 4 - jr; z = nr * nr / 3 - 1; }
    else { nr = 1; z = (2 - jr) * 2 / 3; }
    var phi = nr > 1e-9 ? (Math.PI / 4) * (JPLL[f] * nr + x - y) / nr : 0;
    return [Math.asin(Math.max(-1, Math.min(1, z))), phi];
  }

  // Polylines in (lat, lon), each with a schedule value: the moment in
  // the cure (0 to 1) when it changes, poles first, scattered by region.
  // Regions repeat every 90 degrees, like the spin per loop.
  function region(lat, lon) {
    var li = Math.floor(lat / STEP), lj = Math.floor(lon / STEP);
    return [li, ((lj % 6) + 6) % 6];
  }
  function schedule(lat, lon) {
    var r = region(lat, lon);
    var centre = (r[0] + 0.5) * STEP;
    return {
      cure: 1 - (Math.abs(centre) / HALF_PI) * 0.55 - 0.45 * hash(r[0] + 3, r[1] + 7),
      relapse: hash(r[0] + 11, r[1] + 5),
    };
  }

  // Short segments with their own schedule, so lines fade piece by piece.
  function segments(points) {
    var out = [];
    for (var i = 0; i + 1 < points.length; i++) {
      var a = points[i], b = points[i + 1];
      var s = schedule((a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
      out.push({ a: a, b: b, cure: s.cure, relapse: s.relapse });
    }
    return out;
  }

  var GRATICULE = [];
  (function () {
    var i, k, pts;
    for (i = -5; i <= 5; i++) { // parallels every 15 degrees
      pts = [];
      for (k = 0; k <= 96; k++) pts.push([i * STEP, (k / 96) * 2 * Math.PI]);
      GRATICULE = GRATICULE.concat(segments(pts));
    }
    for (i = 0; i < 24; i++) { // meridians, almost to the poles
      pts = [];
      for (k = 0; k <= 48; k++) pts.push([-HALF_PI * 0.985 + (k / 48) * Math.PI * 0.985, i * STEP]);
      GRATICULE = GRATICULE.concat(segments(pts));
    }
  })();

  // HEALPix edges at nside = 1, 2, 4: lines x = k/n and y = k/n on every face.
  var HEALPIX = [1, 2, 4].map(function (n) {
    var segs = [];
    for (var f = 0; f < 12; f++) {
      for (var k = 0; k <= n; k++) {
        if (n > 1 && (k === 0 || k === n)) continue; // drawn by coarser levels
        var u = k / n, px = [], py = [];
        for (var s = 0; s <= 24; s++) {
          var v = s / 24;
          px.push(faceToLatLon(f, u, v));
          py.push(faceToLatLon(f, v, u));
        }
        segs = segs.concat(segments(px), segments(py));
      }
    }
    return segs;
  });

  // Level-2 cells as outlines, for the healed glow.
  var CELLS = [];
  (function () {
    for (var f = 0; f < 12; f++) {
      for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 4; j++) {
          if (hash(f * 16 + i * 4 + j, 5) < 0.72) continue;
          var ring = [], s;
          for (s = 0; s < 6; s++) ring.push(faceToLatLon(f, (i + s / 6) / 4, j / 4));
          for (s = 0; s < 6; s++) ring.push(faceToLatLon(f, (i + 1) / 4, (j + s / 6) / 4));
          for (s = 0; s < 6; s++) ring.push(faceToLatLon(f, (i + 1 - s / 6) / 4, (j + 1) / 4));
          for (s = 0; s < 6; s++) ring.push(faceToLatLon(f, i / 4, (j + 1 - s / 6) / 4));
          var c = faceToLatLon(f, (i + 0.5) / 4, (j + 0.5) / 4);
          var sc = schedule(c[0], c[1]);
          CELLS.push({ ring: ring, cure: sc.cure, relapse: sc.relapse, delay: hash(f, i * 4 + j) });
        }
      }
    }
  })();

  // Fever: some lat/lon cells near the poles.
  var FEVER = [];
  (function () {
    for (var li = -6; li < 6; li++) {
      for (var lj = 0; lj < 24; lj++) {
        var lat0 = li * STEP, polar = clamp01((Math.abs(lat0 + STEP / 2) - 0.95) / 0.45);
        if (hash(li, lj % 6) < 0.93 - 0.4 * polar) continue;
        var ring = [], s;
        var la = Math.max(lat0, -HALF_PI * 0.985), lb = Math.min(lat0 + STEP, HALF_PI * 0.985);
        for (s = 0; s <= 4; s++) ring.push([la, (lj + s / 4) * STEP]);
        for (s = 0; s <= 4; s++) ring.push([lb, (lj + 1 - s / 4) * STEP]);
        var sc = schedule(lat0 + STEP / 2, (lj + 0.5) * STEP);
        FEVER.push({ ring: ring, cure: sc.cure, relapse: sc.relapse, beat: hash(lj % 6, li) });
      }
    }
  })();

  // ------------------------------------------------------------ drawing
  function makeProjector(spin, cx, cy, r) {
    var ct = Math.cos(TILT), st = Math.sin(TILT);
    return function (p) {
      var lat = p[0], lon = p[1] - spin;
      var gx = Math.cos(lat) * Math.sin(lon), gy = Math.sin(lat), gz = Math.cos(lat) * Math.cos(lon);
      var y = gy * ct - gz * st, z = gy * st + gz * ct;
      return [cx + r * gx, cy - r * y, z];
    };
  }

  // Weight of a part of the drawing at the moment t: how cured it is
  // (0 = lat/lon, 1 = HEALPix).
  function cured(item, front, back) {
    return smooth((front - item.cure) / 0.18) * (1 - smooth((back - item.relapse) / 0.18));
  }

  // Stroke segments, grouped into a few alpha levels to keep it fast.
  function strokeSegments(ctx, segs, project, colour, width, alphaOf) {
    var buckets = [];
    for (var i = 0; i < segs.length; i++) {
      var s = segs[i], a0 = project(s.a), a1 = project(s.b);
      if (a0[2] <= 0 || a1[2] <= 0) continue; // back of the globe
      var alpha = alphaOf(s) * Math.sqrt(Math.min(a0[2], a1[2]));
      if (alpha < 0.02) continue;
      var b = Math.min(15, Math.floor(alpha * 16));
      (buckets[b] = buckets[b] || []).push(a0, a1);
    }
    ctx.lineWidth = width;
    for (var k = 0; k < 16; k++) {
      var pts = buckets[k];
      if (!pts) continue;
      ctx.strokeStyle = rgba(colour, (k + 0.5) / 16);
      ctx.beginPath();
      for (var j = 0; j < pts.length; j += 2) {
        ctx.moveTo(pts[j][0], pts[j][1]);
        ctx.lineTo(pts[j + 1][0], pts[j + 1][1]);
      }
      ctx.stroke();
    }
  }

  function fillRing(ctx, ring, project, style) {
    var pts = ring.map(project);
    for (var i = 0; i < pts.length; i++) if (pts[i][2] <= 0.02) return;
    ctx.fillStyle = style;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (var j = 1; j < pts.length; j++) ctx.lineTo(pts[j][0], pts[j][1]);
    ctx.closePath();
    ctx.fill();
  }

  function render(ctx, w, h, t, pal) {
    ctx.clearRect(0, 0, w, h);
    var r = Math.min(w, h) * 0.44, cx = w / 2, cy = h / 2;
    var project = makeProjector(HALF_PI * (t / LOOP), cx, cy, r);
    var front = 1.25 * smooth(phase(t, 2.0, 4.6));
    var back = 1.25 * smooth(phase(t, 7.2, 8.0));
    var lvl1 = smooth(phase(t, 4.2, 5.0)), lvl2 = smooth(phase(t, 5.2, 6.0));
    var glow = Math.sin(Math.PI * phase(t, 6.0, 7.4));
    var heal = front - back > 0 ? clamp01(front - back) : 0;
    var unit = r / 160; // line widths scale with the globe

    // The globe: soft shading for volume, tinted as it heals.
    var tint = function (c, k) {
      return [0, 1, 2].map(function (i) { return Math.round(c[i] + (pal.ink[i] - c[i]) * k); });
    };
    var g = ctx.createRadialGradient(cx - r * 0.35, cy - r * 0.4, r * 0.1, cx, cy, r);
    g.addColorStop(0, rgba(tint(pal.bodyIn, 0.06 * heal), 1));
    g.addColorStop(1, rgba(tint(pal.bodyOut, 0.04 * heal), 1));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fill();

    // Fever cells, cooling as they are cured.
    for (var i = 0; i < FEVER.length; i++) {
      var fv = FEVER[i], sick = 1 - cured(fv, front, back);
      var pulse = 0.55 + 0.45 * Math.sin(2 * Math.PI * (t * 0.75 + fv.beat));
      if (sick * pulse > 0.03) fillRing(ctx, fv.ring, project, rgba(pal.fever, 0.55 * sick * pulse));
    }

    // Glowing cells once healed.
    if (glow > 0) {
      for (var c = 0; c < CELLS.length; c++) {
        var cell = CELLS[c];
        var a = glow * cured(cell, front, back) * (0.6 + 0.4 * Math.sin(Math.PI * clamp01(glow * 1.4 - cell.delay * 0.4)));
        if (a > 0.02) fillRing(ctx, cell.ring, project, rgba(pal.ink, 0.28 * a));
      }
    }

    ctx.lineCap = "round";
    // The lat/lon grid, redder toward the poles while sick.
    strokeSegments(ctx, GRATICULE.filter(function (s) { return Math.abs(s.a[0]) < 1.0; }),
      project, pal.sick, 1.3 * unit, function (s) { return 0.9 * (1 - cured(s, front, back)); });
    strokeSegments(ctx, GRATICULE.filter(function (s) { return Math.abs(s.a[0]) >= 1.0; }),
      project, [Math.round((pal.sick[0] + pal.fever[0]) / 2), Math.round((pal.sick[1] + pal.fever[1]) / 2), Math.round((pal.sick[2] + pal.fever[2]) / 2)],
      1.3 * unit, function (s) { return 0.9 * (1 - cured(s, front, back)); });

    // HEALPix: base cells, then levels 1 and 2.
    strokeSegments(ctx, HEALPIX[2], project, pal.ink, 1.0 * unit,
      function (s) { return 0.45 * lvl2 * cured(s, front, back); });
    strokeSegments(ctx, HEALPIX[1], project, pal.ink, 1.5 * unit,
      function (s) { return 0.7 * lvl1 * cured(s, front, back); });
    strokeSegments(ctx, HEALPIX[0], project, pal.ink, 2.4 * unit,
      function (s) { return cured(s, front, back); });

    // A faint rim.
    ctx.lineWidth = 1.2 * unit;
    ctx.strokeStyle = rgba(pal.rim, 0.35);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.stroke();
  }

  // ------------------------------------------------------------ driver
  function start(canvas) {
    var ctx = canvas.getContext("2d");
    var still = window.matchMedia("(prefers-reduced-motion: reduce)");
    var visible = true, t0 = null, now = 0, frozen = null, w = 0, h = 0;

    function resize() {
      var dpr = window.devicePixelRatio || 1;
      w = canvas.clientWidth; h = canvas.clientHeight;
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    function current() {
      if (frozen !== null) return frozen;
      return still.matches ? STILL : now;
    }
    function draw(t) { if (w && h) render(ctx, w, h, t, PALETTES[schemeOf(canvas)]); }

    function tick(ms) {
      requestAnimationFrame(tick);
      if (frozen !== null || !visible || document.hidden) { t0 = null; return; }
      if (still.matches) { draw(STILL); return; }
      if (t0 === null) t0 = ms - now * 1000;
      now = ((ms - t0) / 1000) % LOOP;
      draw(now);
    }

    resize();
    if ("ResizeObserver" in window) {
      new ResizeObserver(function () { resize(); draw(current()); }).observe(canvas);
    }
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
      }).observe(canvas);
    }
    // Redraw at once when the reader switches between light and dark.
    new MutationObserver(function () { draw(current()); })
      .observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"], subtree: true });

    // For previews: canvas.gdCure.seek(t) freezes the animation at t,
    // canvas.gdCure.play() resumes it.
    canvas.gdCure = {
      seek: function (t) { frozen = t; now = t; draw(t); },
      play: function () { frozen = null; t0 = null; },
      still: STILL,
    };
    draw(still.matches ? STILL : 0);
    requestAnimationFrame(tick);
  }

  function init() {
    var canvases = document.querySelectorAll("canvas[data-gd-cure]");
    for (var i = 0; i < canvases.length; i++) {
      if (!canvases[i].gdCure) start(canvases[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  // Zensical's instant navigation replaces the page content without a reload.
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);
})();
