export default {
    catalogUrl: "https://freva.dkrz.de/api/freva-nextgen/stacapi/product/?visible_collections=cmip6,dyamond,eerie,icdc,icon-dream,nextgems,orchestra",
  
    // Header
    catalogTitle: "Waterpark STAC Catalogue",
    catalogImage: "https://waterpark.dkrz.de/assets/logo-512.png",
  
    // Footer
    footerLinks: [
      { label: "Freva", url: "https://www.freva.dkrz.de" },
      { label: "DKRZ", url: "https://www.dkrz.de" }
    ],
  
    // Landing content (root catalog only)
    preprocessSTAC(stac, state) {
        const base = u =>
          (u || "")
            .split("?")[0]
            .replace(/\/+$/, "");
      
        const root = base(state.catalogUrl);
        const self = base(typeof stac.getAbsoluteUrl === "function" ? stac.getAbsoluteUrl() : "");
        // only the landing/root catalog
        if (!root || self !== root) return stac;
      
        stac.title = "AI-/Analysis-Ready Cloud Optimized (ARCO) climate data";
        stac.description = [
          "![what is waterpark](/assets/stacapi_anim.gif)",
          "",
          "> Curated Earth-system datasets, remapped to a uniform **HEALPix** grid and stored as cloud-native **Zarr** on **S3** for fast, scalable, server-side analysis."
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
