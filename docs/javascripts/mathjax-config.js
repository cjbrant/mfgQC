// Render math on both the mkdocs pages (arithmatex emits \(...\)) and the
// notebook pages (mkdocs-jupyter leaves raw $...$ in markdown cells).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  },
};
