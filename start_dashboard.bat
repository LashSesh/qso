@echo off
REM ========================================
REM Q⊗DASH Dashboard Starter für Windows
REM ========================================

echo.
echo  ██████╗ ⊗ ██████╗  █████╗ ███████╗██╗  ██╗
echo ██╔═══██╗  ██╔══██╗██╔══██╗██╔════╝██║  ██║
echo ██║   ██║  ██║  ██║███████║███████╗███████║
echo ██║▄▄ ██║  ██║  ██║██╔══██║╚════██║██╔══██║
echo  ╚██████╔╝  ██████╔╝██║  ██║███████║██║  ██║
echo   ╚══▀▀═╝   ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
echo.
echo ========================================
echo  Q⊗DASH Dashboard Starter
echo ========================================
echo.

REM Prüfe ob Rust installiert ist
where cargo >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ FEHLER: Rust/Cargo ist nicht installiert!
    echo.
    echo Bitte installiere erst Rust:
    echo 1. Gehe zu: https://rustup.rs
    echo 2. Lade rustup-init.exe herunter
    echo 3. Installiere Rust
    echo 4. Starte dieses Script nochmal
    echo.
    pause
    exit /b 1
)

echo ✅ Rust gefunden:
cargo --version
echo.

REM Prüfe ob wir im richtigen Ordner sind
if not exist "Cargo.toml" (
    echo ❌ FEHLER: Cargo.toml nicht gefunden!
    echo.
    echo Dieses Script muss im Q⊗DASH Hauptordner ausgeführt werden!
    echo Aktueller Ordner: %CD%
    echo.
    echo Bitte:
    echo 1. Öffne den Datei-Explorer
    echo 2. Gehe zum qdash Ordner
    echo 3. Doppelklick auf start_dashboard.bat
    echo.
    pause
    exit /b 1
)

echo ✅ Richtiger Ordner gefunden
echo Ordner: %CD%
echo.

REM Frage ob das Projekt gebaut werden soll
echo ⚠️  ERSTE VERWENDUNG?
echo.
echo Wenn du das Programm zum ersten Mal startest,
echo musst du es erst bauen (dauert 15-20 Minuten).
echo.
set /p BUILD="Projekt jetzt bauen? (j/n): "

if /i "%BUILD%"=="j" (
    echo.
    echo 🔨 Baue Projekt...
    echo Das kann beim ersten Mal 15-20 Minuten dauern!
    echo ⏳ Bitte warten...
    echo.
    cargo build --workspace --release

    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ❌ FEHLER beim Bauen!
        echo.
        echo Häufige Probleme:
        echo - Visual Studio Build Tools fehlen
        echo - Nicht genug Speicherplatz
        echo - Internet-Verbindung unterbrochen
        echo.
        echo Siehe: docs/WINDOWS_SETUP_DEUTSCH.md
        echo.
        pause
        exit /b 1
    )

    echo.
    echo ✅ Projekt erfolgreich gebaut!
    echo.
)

echo 🚀 Starte Q⊗DASH Dashboard Server...
echo.
echo ⚠️  WICHTIG:
echo - Dieses Fenster NICHT schließen!
echo - Das Dashboard öffnet sich gleich im Browser
echo - Zum Beenden: Drücke Strg+C in diesem Fenster
echo.
echo ========================================
echo.

REM Warte 3 Sekunden
timeout /t 3 /nobreak >nul

REM Öffne Browser nach 5 Sekunden im Hintergrund
start /B cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:8080"

REM Starte den Server
cargo run --bin metatron_telemetry --release

REM Wenn wir hier ankommen, wurde der Server gestoppt
echo.
echo ========================================
echo  Server wurde beendet.
echo ========================================
echo.
pause
