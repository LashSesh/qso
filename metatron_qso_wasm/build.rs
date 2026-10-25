// The canonical labelling of large graphs recurses once per node; give the
// WebAssembly module an 8 MiB stack instead of the linker's 1 MiB.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("wasm") {
        println!("cargo:rustc-link-arg-cdylib=-zstack-size=8388608");
    }
}
