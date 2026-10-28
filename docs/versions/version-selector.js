// Version selector of the documentation site. docs/build_site.py copies this
// file to the site root, writes versions.json next to it ([{version, path}],
// the development version first) and loads it on every page of the guides.
// It adds a <select> to mdBook's menu bar; choosing a version (with the
// mouse, or with the arrow keys and Enter) opens the same page in that
// version, or its start page if the page is missing there.
(function () {
  "use strict";
  const script = document.currentScript;
  if (!script) return;
  const root = new URL(".", script.src);
  const base = (v) => new URL(v.path, root).href;
  fetch(new URL("versions.json", root))
    .then((r) => r.json())
    .then((versions) => {
      const here = window.location.href.split("#")[0].split("?")[0];
      let current = versions[0];
      for (const v of versions) {
        if (here.startsWith(base(v)) && base(v).length > base(current).length) current = v;
      }
      const page = here.slice(base(current).length) || "index.html";
      const select = document.createElement("select");
      select.className = "qso-version-selector";
      select.setAttribute("aria-label", "Documentation version");
      for (const v of versions) {
        const option = document.createElement("option");
        option.value = v.path;
        option.textContent = v.version;
        option.selected = v === current;
        select.appendChild(option);
      }
      const go = () => {
        if (select.value === current.path) return;
        const start = new URL(select.value, root);
        const target = new URL(page, start);
        // Keep the query and the anchor when the page exists there.
        fetch(target, { method: "HEAD" })
          .then((r) => {
            window.location.href = r.ok ? target.href + window.location.search + window.location.hash : start.href;
          })
          .catch(() => { window.location.href = start.href; });
      };
      // Arrow keys change a closed select's value: with the keyboard, only
      // Enter switches the version, so the list can be stepped through.
      let keyboard = false;
      select.addEventListener("keydown", (e) => {
        keyboard = e.key !== "Enter";
        if (e.key === "Enter") go();
      });
      select.addEventListener("pointerdown", () => { keyboard = false; });
      select.addEventListener("change", () => { if (!keyboard) go(); });
      // Leaving without Enter keeps the current version selected.
      select.addEventListener("blur", () => { select.value = current.path; });
      const bar = document.querySelector("#mdbook-menu-bar .right-buttons, #menu-bar .right-buttons");
      if (bar) {
        select.style.cssText = "margin: 0 0.5em; font-size: 0.8em; color: var(--fg); background: var(--bg); border: 1px solid var(--icons); border-radius: 4px;";
        bar.prepend(select);
      }
    })
    .catch(() => {});
})();
