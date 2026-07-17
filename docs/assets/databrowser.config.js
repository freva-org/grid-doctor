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

  // Optional freva-web-parity script that maps facet VALUES to human-readable
  // descriptions.
  metadataScriptUrl: "",

  // Public, read-only demo
  authEnabled: false,
  enableHeavyOps: false,

  // Mirror the active query (flavour + facets + time + bbox) into the page URL
  // so a link reproduces the exact view, and read it back on load. Safe here
  syncUrl: true,

  // Scope: Waterpark-only "hosted filtered instance"
  baseFilters: { project: "waterpark" },

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
