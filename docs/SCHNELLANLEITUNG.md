# ⚡ Q⊗DASH Schnellanleitung für Windows 10

## 📋 Checkliste - Hake ab, wenn fertig!

- [ ] **Schritt 1**: ZIP heruntergeladen und entpackt
- [ ] **Schritt 2**: Rust installiert
- [ ] **Schritt 3**: Visual Studio Build Tools installiert
- [ ] **Schritt 4**: Projekt gebaut
- [ ] **Schritt 5**: Server gestartet
- [ ] **Schritt 6**: Dashboard im Browser geöffnet

---

## 🎯 Die 6 Schritte im Überblick

### 1️⃣ DOWNLOAD & ENTPACKEN (5 Minuten)
```
🌐 Browser → github.com/LashSesh/qdash
📥 Grüner Button "Code" → "Download ZIP"
📦 Rechtsklick auf ZIP → "Alle extrahieren" → Desktop\qdash
```

### 2️⃣ RUST INSTALLIEREN (10 Minuten)
```
🌐 Browser → rustup.rs
📥 "rustup-init.exe" herunterladen
▶️ Doppelklick → Enter drücken → Warten
✅ Test: Windows-Taste → cmd → rustc --version
```

### 3️⃣ BUILD TOOLS INSTALLIEREN (30-40 Minuten)
```
🌐 Browser → visualstudio.microsoft.com/downloads
📥 "Build Tools für Visual Studio 2022"
▶️ Doppelklick → "Desktop-Entwicklung mit C++" ankreuzen ✓
⏳ Installieren → Warten (lange!) → Schließen
```

### 4️⃣ PROJEKT BAUEN (15-20 Minuten)
```
📁 Datei-Explorer → Desktop\qdash
📌 Adressleiste anklicken → "cmd" tippen → Enter
⚙️ Im schwarzen Fenster:
   cargo build --workspace --release
⏳ Warten bis "Finished release..." erscheint
```

### 5️⃣ SERVER STARTEN (30 Sekunden)
```
⚙️ Im schwarzen Fenster:
   cargo run --bin metatron_telemetry --release
⏳ Warten bis "Listening on http://0.0.0.0:8080"
✅ Fenster NICHT schließen!
```

### 6️⃣ DASHBOARD ÖFFNEN (10 Sekunden)
```
🌐 Browser → http://localhost:8080
🎉 Dashboard sollte erscheinen!
```

---

## 🎮 Schnellbefehle

| Was                | Befehl                                           |
|--------------------|--------------------------------------------------|
| **Server starten** | `cargo run --bin metatron_telemetry --release`  |
| **Server stoppen** | `Strg + C` im schwarzen Fenster                 |
| **Projekt bauen**  | `cargo build --workspace --release`             |
| **Dashboard URL**  | `http://localhost:8080`                         |
| **Rust prüfen**    | `rustc --version`                               |
| **Zum Ordner**     | `cd C:\Users\DEINNAME\Desktop\qdash`            |

---

## 🐛 3 Häufigste Fehler

### ❌ "rustc nicht erkannt"
```
✅ Kommandozeile neu öffnen
✅ Computer neustarten
✅ Rust nochmal installieren (rustup.rs)
```

### ❌ "linker 'link.exe' not found"
```
✅ Visual Studio Build Tools installieren
✅ "Desktop-Entwicklung mit C++" auswählen ✓
✅ Computer neustarten
```

### ❌ Dashboard zeigt "Cannot connect"
```
✅ Server läuft nicht → Schritt 5 wiederholen
✅ Warte 10 Sekunden nach dem Start
✅ Prüfe: "Listening on..." steht im schwarzen Fenster?
```

---

## 💾 Dateigröße & Zeit

| Was                 | Größe  | Zeit        |
|---------------------|--------|-------------|
| ZIP Download        | ~50 MB | 1-2 Min     |
| Rust Installation   | ~300 MB| 5-10 Min    |
| Build Tools         | ~6 GB  | 30-40 Min   |
| Projekt Build       | ~2 GB  | 15-20 Min   |
| **GESAMT**          | ~8 GB  | **~60 Min** |

---

## 🎯 Systemanforderungen

| Komponente      | Minimum           | Empfohlen         |
|-----------------|-------------------|-------------------|
| **OS**          | Windows 10 64-bit | Windows 10/11     |
| **RAM**         | 4 GB              | 8 GB              |
| **Festplatte**  | 10 GB frei        | 20 GB frei        |
| **Prozessor**   | Dual-Core         | Quad-Core         |
| **Internet**    | Für Installation  | Für Installation  |

---

## 🔧 Konfiguration (Optional)

### Standard (Sicher - Nur Simulation):
```
Keine Konfiguration nötig!
Einfach starten und loslegen.
```

### IBM Dry-Run Mode (Test ohne echten Quantum-Computer):
```
1. Kopiere .env.example → .env
2. Öffne .env mit Notepad
3. Ändere: IBM_BACKEND_MODE=dry-run
4. Speichern & Server neu starten
```

---

## 📱 Dashboard Übersicht

```
┌─────────────────────────────────────────────────────────┐
│  Q⊗DASH - Metatron VM                                   │
│  Quantum-Hybrid Calibration System                      │
└─────────────────────────────────────────────────────────┘

┌────────────────┬────────────────┐
│ System Status  │  Recent Jobs   │
│                │                │
│ ψ: 0.8500      │ Job #abc123    │
│ ρ: 0.9000      │ [COMPLETED]    │
│ ω: 0.7500      │                │
│                │ Job #def456    │
│ Backend:       │ [RUNNING]      │
│ [SIMULATOR]    │                │
│ local_sim      │                │
│ 13 qubits      │                │
├────────────────┼────────────────┤
│ Metrics Chart  │ Control Panel  │
│                │                │
│  [Graph mit    │ ▶ Start        │
│   3 Linien]    │   Calibration  │
│                │                │
│                │ 🔄 Refresh All │
└────────────────┴────────────────┘
```

---

## 🎮 Erste Schritte nach dem Start

1. **Beobachte die Metriken**
   - Die Zahlen (ψ, ρ, ω) aktualisieren sich alle 5 Sekunden
   - Die drei grünen Punkte (●) zeigen: Alles läuft!

2. **Starte eine Kalibrierung**
   - Klicke "Start Calibration" ▶
   - Warte ein paar Sekunden
   - Schau zu, wie ein neuer Job in "Recent Jobs" erscheint

3. **Beobachte das Diagramm**
   - Die Linien zeigen die Metrik-Historie
   - Grün = ψ (Quality)
   - Blau = ρ (Stability)
   - Orange = ω (Efficiency)

---

## 🚪 Beenden & Neustarten

### Beenden:
```
1. Server-Fenster (schwarz) → Strg + C
2. Browser schließen (optional)
3. Fertig!
```

### Neustarten:
```
1. Datei-Explorer → Desktop\qdash
2. Adressleiste → "cmd"
3. cargo run --bin metatron_telemetry --release
4. Browser → http://localhost:8080
```

---

## 📞 Hilfe & Kontakt

| Problem                    | Wo findest du Hilfe?                      |
|----------------------------|-------------------------------------------|
| **Detaillierte Anleitung** | `docs/WINDOWS_SETUP_DEUTSCH.md`          |
| **Backend Info**           | `docs/backend_system.md` (Englisch)      |
| **Fehlermeldungen**        | Kopiere die Fehlermeldung → Google       |
| **GitHub Issues**          | github.com/LashSesh/qdash/issues         |

---

## ⚡ Pro-Tipps

1. **Speichere die Server-Start-Befehle**
   - Erstelle eine Textdatei mit den Befehlen
   - Einfach copy-paste beim nächsten Mal

2. **Lesezeichen setzen**
   - Speichere `http://localhost:8080` als Lesezeichen
   - Schneller Zugriff beim nächsten Mal

3. **Server im Hintergrund**
   - Minimiere das schwarze Fenster (nicht schließen!)
   - Server läuft weiter, du kannst andere Sachen machen

4. **Performance**
   - Schließe andere Programme beim ersten Build
   - Mehr RAM = Schnellerer Build

---

## 🏁 Fertig!

**Du bist bereit für dein Q⊗DASH Abenteuer!** 🚀

```
     🎮 Dashboard läuft
     ✅ Alles funktioniert
     🚀 Viel Spaß!
```

---

## 📅 Versions-Info

- **Letzte Aktualisierung**: 2025
- **Getestet auf**: Windows 10 64-bit
- **Q⊗DASH Version**: 0.1.0
- **Rust Version**: 1.75+ (oder neuer)

---

## 🌈 Bonus: ASCII Art für's Dashboard

Wenn alles läuft, solltest du das sehen:

```
  ██████╗ ⊗ ██████╗  █████╗ ███████╗██╗  ██╗
 ██╔═══██╗  ██╔══██╗██╔══██╗██╔════╝██║  ██║
 ██║   ██║  ██║  ██║███████║███████╗███████║
 ██║▄▄ ██║  ██║  ██║██╔══██║╚════██║██╔══██║
 ╚██████╔╝  ██████╔╝██║  ██║███████║██║  ██║
  ╚══▀▀═╝   ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
```

**Viel Erfolg!** 🎉
