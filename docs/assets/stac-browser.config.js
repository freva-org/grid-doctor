export default {
    catalogUrl: "https://freva.dkrz.de/api/freva-nextgen/stacapi/product/?visible_collections=cmip6,dyamond,eerie,icdc,icon-dream,nextgems,obs,palmod,reanalysis-healpix",

    // Header
    catalogTitle: "STAC Browser",
    catalogImage: "https://waterpark.dkrz.de/assets/logo-512.png",

    // Footer
    footerLinks: [
      { label: "Freva", url: "https://www.freva.dkrz.de" },
      { label: "DKRZ", url: "https://www.dkrz.de" }
    ],

    // Landing content (root catalog only)
    preprocessSTAC(stac, state) {
        // Canonicalize root-equivalent links to catalogUrl so "up" lands on "/" not /external/...
        const catalog = (() => { try { return new URL(state.catalogUrl); } catch { return null; } })();
        if (catalog && Array.isArray(stac.links)) {
          const stripSlash = p => p.replace(/\/+$/, "");
          const rootPath = stripSlash(catalog.pathname);
          for (const link of stac.links) {
            if (!link || typeof link.href !== "string") continue;
            let u;
            try { u = new URL(link.href, state.catalogUrl); } catch { continue; }
            if (u.host === catalog.host && stripSlash(u.pathname) === rootPath) {
              link.href = state.catalogUrl;
            }
          }
        }

        const base = u =>
          (u || "")
            .split("?")[0]
            .replace(/\/+$/, "");

        const root = base(state.catalogUrl);
        const self = base(typeof stac.getAbsoluteUrl === "function" ? stac.getAbsoluteUrl() : "");
        // only the landing/root catalog
        if (!root || self !== root) return stac;

        stac.title = "AI-/Analysis-Ready Cloud Optimized";
        stac.description = [
          "![what is waterpark](/assets/stacapi_anim.gif)",
          "",
          "> Explore the HEALPix data hub with our STAC Browser."
        ].join("\n");
        stac.keywords = ["climate", "HEALPix", "Zarr", "FAIR", "Freva"];
        stac.license = "CC-BY-4.0";
        stac.providers = [
          { name: "Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research (AWI)", roles: ["producer"], url: "https://www.awi.de/en/" },
          { name: "Helmholtz-Zentrum Hereon", roles: ["producer"], url: "https://www.hereon.de/" },
          { name: "Universität Hamburg", roles: ["producer"], url: "https://www.uni-hamburg.de/en.html" },
          { name: "Max Planck Institute for Meteorology (MPI-M)", roles: ["producer"], url: "https://mpimet.mpg.de/en/homepage" },
          { name: "Deutsches Klimarechenzentrum (DKRZ)", roles: ["host"], url: "https://www.dkrz.de" }
        ];

        if (Array.isArray(stac.links)) {
          stac.links = stac.links.filter(l => ((l && l.rel) || "").toLowerCase() !== "service-desc");
        }
        return stac;
      },
  };
