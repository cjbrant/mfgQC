// Clicking a header's link icon copies the section's deep link to the clipboard
// instead of just jumping to the anchor. Delegated so it survives re-renders.
document.addEventListener("click", function (e) {
  var a = e.target.closest(".md-typeset a.headerlink");
  if (!a) return;
  e.preventDefault();
  var href = a.getAttribute("href");
  var url = location.origin + location.pathname + href;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(url);
  }
  try {
    history.replaceState(null, "", href);
  } catch (err) {}
  a.classList.add("bs-copied");
  window.setTimeout(function () {
    a.classList.remove("bs-copied");
  }, 1200);
});
