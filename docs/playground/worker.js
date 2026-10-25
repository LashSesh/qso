// Runs QSO off the page's main thread: {id, method, params} in,
// {id, ok} or {id, error} out.
import { load } from "./qso/index.js";

const ready = load(new URL("./qso/qso.wasm", import.meta.url));

self.onmessage = async ({ data: { id, method, params } }) => {
  try {
    const qso = await ready;
    self.postMessage({ id, ok: qso.call(method, params) });
  } catch (e) {
    self.postMessage({ id, error: e.message });
  }
};
