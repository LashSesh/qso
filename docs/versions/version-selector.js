// Version selector of the documentation site. docs/build_site.py copies this
// file to the site root, writes versions.json next to it ([{version, path}],
// the development version first) and loads it on every page of the guides.
// It adds a <select> to mdBook's menu bar; switching opens the same page in
// the other version, or that version's start page if the page is missing.
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
      select.addEventListener("change", () => {
        const target = new URL(page, new URL(select.value, root));
        fetch(target, { method: "HEAD" })
          .then((r) => { window.location.href = r.ok ? target.href : new URL(select.value, root).href; })
          .catch(() => { window.location.href = new URL(select.value, root).href; });
      });
      const bar = document.querySelector("#mdbook-menu-bar .right-buttons, #menu-bar .right-buttons");
      if (bar) {
        select.style.cssText = "margin: 0 0.5em; font-size: 0.8em; color: var(--fg); background: var(--bg); border: 1px solid var(--icons); border-radius: 4px;";
        bar.prepend(select);
      }
    })
    .catch(() => {});
})();
