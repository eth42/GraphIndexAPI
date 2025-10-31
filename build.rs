use std::env;

fn main() {
	use std::path::Path;
	let rust_toolchain = env::var("RUSTUP_TOOLCHAIN").unwrap();
	let rust_toolchain = Path::new(rust_toolchain.as_str()).file_name().unwrap().to_str().unwrap();
	if rust_toolchain.starts_with("stable") {
		// do nothing
	} else if rust_toolchain.starts_with("1.86.0") {
		// do nothing
	} else if rust_toolchain.starts_with("nightly") {
		//enable the 'nightly-features' feature flag
		println!("cargo:rustc-cfg=feature=\"nightly-features\"");
	} else {
		panic!("Unexpected value for rustc toolchain {:?}", rust_toolchain)
	}
}
