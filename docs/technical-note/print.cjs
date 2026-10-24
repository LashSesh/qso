// Print an HTML file to an A4 PDF with headless Chromium (Playwright).
// Usage: node print.cjs input.html output.pdf
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const [input, output] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.resolve(input), { waitUntil: "load" });
  await page.pdf({
    path: output,
    format: "A4",
    printBackground: true,
    margin: { top: "20mm", bottom: "20mm", left: "19mm", right: "19mm" },
    displayHeaderFooter: true,
    headerTemplate: "<span></span>",
    footerTemplate:
      '<div style="width:100%;text-align:center;font-size:8pt;color:#666;' +
      'font-family:Charter,\'Bitstream Charter\',serif">' +
      '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
  });
  await browser.close();
})();
