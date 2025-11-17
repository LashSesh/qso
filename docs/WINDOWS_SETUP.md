# Windows Setup Guide for Q⊗DASH

Complete guide for setting up the Q⊗DASH quantum computing framework on Windows 10/11.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Required Tools Installation](#required-tools-installation)
3. [Building the Project](#building-the-project)
4. [Troubleshooting](#troubleshooting)
5. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **Operating System**: Windows 10 or Windows 11
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk Space**: At least 5 GB free
- **Internet Connection**: Required for downloading dependencies

### Time Estimate

- Total setup time: 40-60 minutes
  - Rust installation: 5-10 minutes
  - Visual Studio Build Tools: 20-40 minutes
  - CMake installation: 5 minutes
  - NASM installation: 5 minutes
  - First build: 10-20 minutes

---

## Required Tools Installation

### 1. Install Rust

Rust is the primary programming language for Q⊗DASH.

#### 1.1 Download Rust

1. Open your web browser
2. Navigate to: https://rustup.rs
3. Click **"Download rustup-init.exe (64-bit)"**
4. Save the file to your Downloads folder

#### 1.2 Run Rust Installer

1. Open your **Downloads** folder
2. Double-click `rustup-init.exe`
3. A command prompt window will appear
4. Press **Enter** to proceed with default installation
5. Wait for installation to complete (5-10 minutes)
6. When you see "Rust is installed now. Great!", press **Enter** to close

#### 1.3 Verify Rust Installation

1. Press **Windows Key**
2. Type: `cmd`
3. Click **"Command Prompt"**
4. In the command prompt, type:
   ```cmd
   rustc --version
   ```
5. You should see output like: `rustc 1.85.0 (or newer)`

**If you see an error**, close and reopen the command prompt, then try again.

---

### 2. Install Visual Studio Build Tools

The Rust compiler requires Microsoft's C++ build tools for Windows.

#### 2.1 Download Build Tools

1. Navigate to: https://visualstudio.microsoft.com/downloads/
2. Scroll down to **"Tools for Visual Studio"**
3. Click **"Build Tools for Visual Studio 2022"**
4. Download `vs_BuildTools.exe`

#### 2.2 Install Build Tools

1. Double-click `vs_BuildTools.exe` from your Downloads
2. The Visual Studio Installer will launch
3. Wait for it to initialize
4. **CRITICAL**: Select **"Desktop development with C++"**
   - Click on the tile/box to check it ✓
5. On the right side, ensure these are selected:
   - MSVC v143 (or latest)
   - Windows 10/11 SDK
   - C++ CMake tools for Windows (this is important!)
6. Click **"Install"** (bottom right)
7. **Wait 20-40 minutes** for installation
8. Click **"Close"** when finished

**Note**: This is a large download (6-7 GB). Make sure you have a stable internet connection.

---

### 3. Install CMake

CMake is required for building cryptographic dependencies (aws-lc-sys).

#### 3.1 Download CMake

1. Navigate to: https://cmake.org/download/
2. Under **"Binary distributions"**, find **"Windows x64 Installer"**
3. Download the `.msi` file (e.g., `cmake-3.28.0-windows-x86_64.msi`)

#### 3.2 Install CMake

1. Double-click the downloaded `.msi` file
2. Click **"Next"**
3. Accept the license agreement
4. **IMPORTANT**: Select **"Add CMake to the system PATH for all users"**
   - This is crucial for Cargo to find CMake
5. Click **"Next"**
6. Click **"Install"**
7. Click **"Finish"**

#### 3.3 Verify CMake Installation

1. **Close and reopen** your command prompt (important!)
2. Type:
   ```cmd
   cmake --version
   ```
3. You should see: `cmake version 3.28.0` (or newer)

**If you see an error**: Restart your computer and try again.

---

### 4. Install NASM (Optional but Recommended)

NASM is an assembler that improves performance of cryptographic operations.

#### 4.1 Download NASM

1. Navigate to: https://www.nasm.us/pub/nasm/releasebuilds/
2. Find the latest version folder (e.g., `2.16.03/`)
3. Click on the folder
4. Download `nasm-2.16.03-installer-x64.exe` (or latest version)

#### 4.2 Install NASM

1. Double-click the downloaded `.exe` file
2. Click **"Next"**
3. Accept the license agreement
4. **IMPORTANT**: Note the installation path (usually `C:\Program Files\NASM`)
5. Click **"Next"** → **"Install"**
6. Click **"Finish"**

#### 4.3 Add NASM to PATH

1. Press **Windows Key**
2. Type: `environment variables`
3. Click **"Edit the system environment variables"**
4. Click **"Environment Variables..."** button (bottom right)
5. Under **"System variables"**, find and select **"Path"**
6. Click **"Edit..."**
7. Click **"New"**
8. Add: `C:\Program Files\NASM`
9. Click **"OK"** on all windows
10. **Restart your command prompt**

#### 4.4 Verify NASM Installation

```cmd
nasm -version
```

You should see: `NASM version 2.16.03` (or newer)

---

## Building the Project

### 1. Download Q⊗DASH

#### Option A: Using Git (Recommended)

If you have Git installed:

```cmd
cd %USERPROFILE%\Desktop
git clone https://github.com/LashSesh/qso.git
cd qso
```

#### Option B: Download ZIP

1. Go to: https://github.com/LashSesh/qso
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract to `C:\Users\YourUsername\Desktop\qso`

### 2. Open Command Prompt in Project Directory

#### Method A (Easy):

1. Open File Explorer
2. Navigate to: `C:\Users\YourUsername\Desktop\qso`
3. Click in the **address bar** at the top
4. Type: `cmd`
5. Press **Enter**

#### Method B (Classic):

1. Press **Windows Key**
2. Type: `cmd`
3. Open **Command Prompt**
4. Type:
   ```cmd
   cd %USERPROFILE%\Desktop\qso
   ```

### 3. Verify You're in the Correct Directory

```cmd
dir
```

You should see:
- `Cargo.toml`
- `metatron-qso-rs`
- `metatron_backend`
- `docs`
- etc.

### 4. Build the Project

```cmd
cargo build --workspace --release
```

**What happens:**
- ✓ Downloads and compiles all dependencies
- ✓ Compiles all 24 workspace crates
- ✓ Creates optimized binaries
- ⏳ **Takes 10-20 minutes on first build**
- 💨 Your computer fan may get loud (normal!)

**Expected output:**
```
   Compiling libc v0.2.177
   Compiling nalgebra v0.33.0
   Compiling metatron-qso-rs v0.1.0
   ...
   Finished release [optimized] target(s) in 15m 32s
```

**If successful**, you'll see: `Finished release [optimized]`

### 5. Run Tests (Optional)

```cmd
cargo test --workspace
```

This verifies everything is working correctly.

---

## Troubleshooting

### Error: "cmake not found" or "Missing dependency: cmake"

**Cause**: CMake is not installed or not in PATH

**Solution**:
1. Verify CMake is installed: `cmake --version`
2. If not found:
   - Install CMake (see section 3)
   - **Restart your command prompt**
   - Try again
3. If still not working:
   - Restart your computer
   - Verify PATH includes CMake (section 3.3)

---

### Error: "NASM command not found or failed to execute"

**Cause**: NASM is not installed (this is a warning, not critical)

**Solution**:
- **Option 1** (Recommended): Install NASM (see section 4)
- **Option 2**: Ignore the warning - the build will still work, just slower

---

### Error: "error: linker 'link.exe' not found"

**Cause**: Visual Studio Build Tools not installed correctly

**Solution**:
1. Reinstall Visual Studio Build Tools (section 2)
2. Ensure **"Desktop development with C++"** is checked ✓
3. Restart your computer
4. Try building again

---

### Error: "edition 2024 is unstable"

**Cause**: Rust version is too old

**Solution**:
```cmd
rustup update
rustc --version
```

Ensure you have Rust 1.85.0 or newer.

---

### Error: "access is denied" or permission errors

**Cause**: Antivirus or Windows Defender blocking Cargo

**Solution**:
1. Add exclusion for `%USERPROFILE%\.cargo` folder
2. Add exclusion for your project folder
3. Try building again

---

### Build is extremely slow

**Causes**:
- NASM not installed
- Antivirus scanning every file
- HDD instead of SSD

**Solutions**:
1. Install NASM (section 4)
2. Add Cargo exclusions to antivirus
3. Close other programs
4. Be patient - first build is always slowest

---

### Error: Port already in use (when running telemetry)

**Solution**:
```cmd
set TELEMETRY_PORT=8081
cargo run --bin metatron_telemetry --release
```

Then open: `http://localhost:8081`

---

## Next Steps

### 1. Run the Telemetry Dashboard

```cmd
cargo run --bin metatron_telemetry --release
```

Then open your browser to: `http://localhost:8080`

### 2. Run Benchmarks

```cmd
cargo run --release --bin quantum_walk_bench
cargo run --release --bin vqe_bench
cargo run --release --bin qaoa_bench
```

### 3. Explore the Python SDK

```cmd
cd metatron_qso_py
pip install maturin
maturin develop --release
python examples/01_quantum_walk_basic.py
```

### 4. Read the Documentation

- [README.md](../README.md) - Project overview
- [DEV_SETUP.md](../DEV_SETUP.md) - Developer guide
- [PRODUCT_OVERVIEW.md](../PRODUCT_OVERVIEW.md) - Architecture

---

## Summary Checklist

Before building, ensure you have:

- [ ] Rust 1.85.0+ installed
- [ ] Visual Studio Build Tools 2022 installed
- [ ] "Desktop development with C++" workload installed
- [ ] CMake installed and in PATH
- [ ] NASM installed and in PATH (optional)
- [ ] Command prompt restarted after installations
- [ ] At least 5 GB free disk space

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/LashSesh/qso/issues)
2. Read the [DEV_SETUP.md](../DEV_SETUP.md) guide
3. Search for the error message online
4. Open a new issue with:
   - Your Windows version
   - Rust version (`rustc --version`)
   - CMake version (`cmake --version`)
   - Full error message

---

## Additional Resources

- **Rust Windows Guide**: https://rust-lang.github.io/rustup/installation/windows.html
- **CMake Documentation**: https://cmake.org/documentation/
- **Visual Studio Downloads**: https://visualstudio.microsoft.com/downloads/
- **NASM Documentation**: https://www.nasm.us/docs.php

---

**Last updated**: 2025-11-17

**Made with ❤️ in Rust** | **© 2025 Sebastian Klemm (Aion-Chronos)**
