/*
 * Freva Data Browser embed configuration.
 *
 * This is the single place to tune the embedded Data Browser.
 */
export default {
  // Backend 
  apiBase: "https://freva.dkrz.de/api/freva-nextgen/databrowser",

  // Default metadata flavour.
  flavour: "waterpark",

  // Public, read-only demo
  authEnabled: false,
  enableHeavyOps: false,

  // Mirror the active query (flavour + facets + time + bbox) into the page URL
  // so a link reproduces the exact view, and read it back on load. Safe here
  syncUrl: true,

  // Scope: Waterpark-only "hosted filtered instance"
  baseFilters: { project: "waterpark" },

  // the scope value renders locked and survives "Clear all". Set true to let
  // a user take it off and see the wider archive.
  scopeRemovable: false,

  // Land on the metadata overview rather than the file list for the first time
  // user. Always a user's own choice wins on every later visit.
  defaultLayout: "overview",

  // How the overview is laid out before a visitor rearranges it.
  overview: {
    order: [
      "product",
      "variable",
      "time_frequency",
      "healpix_level",
      "experiment",
      "model",
      "realm",
      "ensemble",
      "time_aggregation",
      "__time",
      "__bbox",
    ],
    // Everything not listed here moves under "Show additional facets".
    mainFacets: [
      "product",
      "variable",
      "time_frequency",
      "healpix_level",
      "experiment",
      "model",
      "realm",
      "ensemble",
      "time_aggregation",
      "__time",
      "__bbox",
    ],
  },

  // Brand
  brand: {
    title: "Freva Data-Browser",
    mark: "≈",
    description: "Search the HEALPix data hub via the freva-nextgen API.",
  },


  // Theme: match the Waterpark dominant colour
  // The widget's OWN day/night toggle is disabled
  features: {
    themeToggle: false,
  },
  theme: {
    both: {
      accent: "var(--md-primary-fg-color, #009688)",
      "accent-2": "var(--md-primary-fg-color--dark, #00695c)",
    },
  },
};
