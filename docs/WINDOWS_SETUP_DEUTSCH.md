# 🎮 Q⊗DASH für Kinder - Windows 10 Anleitung

## 📋 Was du brauchst

- Einen Windows 10 Computer
- Internet-Verbindung
- Ungefähr 30 Minuten Zeit
- Mindestens 2 GB freien Speicherplatz

---

## 🎯 Schritt 1: Lade das Projekt herunter

### 1.1 Öffne deinen Internet-Browser (z.B. Chrome, Firefox, Edge)

### 1.2 Gehe zu GitHub
- Tippe diese Adresse ein: `https://github.com/LashSesh/qdash`
- Drücke Enter

### 1.3 Lade das ZIP herunter
1. Suche den **grünen Button** mit dem Text "Code"
2. Klicke darauf
3. Klicke auf **"Download ZIP"**
4. Warte, bis der Download fertig ist (du siehst unten im Browser eine Datei namens `qdash-main.zip`)

### 1.4 Finde die heruntergeladene Datei
1. Öffne den **Downloads-Ordner**
   - Drücke die Windows-Taste + E (öffnet den Datei-Explorer)
   - Klicke links auf "Downloads"
2. Du solltest jetzt `qdash-main.zip` sehen

---

## 📦 Schritt 2: Entpacke das ZIP-File

### 2.1 Entpacke auf dem Desktop
1. **Rechtsklick** auf `qdash-main.zip`
2. Wähle **"Alle extrahieren..."**
3. Ein Fenster öffnet sich
4. Ändere das Ziel zu: `C:\Users\DEINNAME\Desktop\qdash`
   - (Ersetze DEINNAME mit deinem Windows-Benutzernamen)
5. Klicke auf **"Extrahieren"**
6. Warte, bis alle Dateien entpackt sind

### 2.2 Prüfe, ob es geklappt hat
1. Auf deinem Desktop sollte jetzt ein Ordner **"qdash"** sein
2. Öffne ihn (Doppelklick)
3. Du solltest viele Ordner und Dateien sehen, zum Beispiel:
   - 📁 metatron-qso-rs
   - 📁 metatron_backend
   - 📁 docs
   - 📄 Cargo.toml
   - 📄 README.md

**✅ Super! Schritt 2 ist fertig!**

---

## 🛠️ Schritt 3: Installiere die benötigten Programme

### 3.1 Installiere Rust (Die Programmiersprache)

#### 3.1.1 Lade Rust herunter
1. Öffne deinen Browser
2. Gehe zu: `https://rustup.rs`
3. Klicke auf den großen Button **"rustup-init.exe (64-bit)"**
4. Warte, bis der Download fertig ist

#### 3.1.2 Installiere Rust
1. Gehe zu deinen Downloads
2. **Doppelklick** auf `rustup-init.exe`
3. Ein schwarzes Fenster (Kommandozeile) öffnet sich
4. Es fragt: "Proceed with installation (default)?"
5. Drücke **Enter** (die große Taste mit dem Pfeil ↵)
6. Warte 5-10 Minuten (es lädt viele Sachen herunter)
7. Wenn es fertig ist, steht da: "Rust is installed now. Great!"
8. Drücke **Enter** zum Schließen

#### 3.1.3 Prüfe, ob Rust funktioniert
1. Drücke die **Windows-Taste**
2. Tippe: `cmd`
3. Klicke auf **"Eingabeaufforderung"** (das schwarze Fenster-Symbol)
4. Ein schwarzes Fenster öffnet sich
5. Tippe: `rustc --version`
6. Drücke **Enter**
7. Du solltest etwas sehen wie: `rustc 1.75.0` (die Zahl kann anders sein)

**✅ Rust ist installiert!**

### 3.2 Installiere Visual Studio C++ Build Tools (Wird von Rust gebraucht)

#### 3.2.1 Lade die Build Tools herunter
1. Gehe zu: `https://visualstudio.microsoft.com/de/downloads/`
2. Scrolle nach unten zu **"Tools für Visual Studio"**
3. Klicke auf **"Build Tools für Visual Studio 2022"** (Download)
4. Warte, bis `vs_BuildTools.exe` heruntergeladen ist

#### 3.2.2 Installiere die Build Tools
1. **Doppelklick** auf `vs_BuildTools.exe` in deinen Downloads
2. Ein Installer-Fenster öffnet sich
3. Warte kurz, bis es startet
4. **WICHTIG**: Wähle **"Desktop-Entwicklung mit C++"**
   - Es ist eine Kachel mit einem C++ Symbol
   - Klicke einmal drauf, damit ein Häkchen erscheint ✓
5. Rechts siehst du, dass ungefähr 6-7 GB heruntergeladen werden
6. Klicke unten rechts auf **"Installieren"**
7. **Gehe eine Pause machen** (das dauert 20-40 Minuten!)
8. Wenn fertig, klicke auf **"Schließen"**

**✅ Build Tools sind installiert!**

---

## 🚀 Schritt 4: Baue das Projekt

### 4.1 Öffne die Kommandozeile im richtigen Ordner

#### Methode A (Einfach):
1. Öffne den Datei-Explorer (Windows-Taste + E)
2. Gehe zu: `C:\Users\DEINNAME\Desktop\qdash`
3. Klicke in die **Adressleiste** oben (wo "Desktop > qdash" steht)
4. Tippe: `cmd`
5. Drücke **Enter**
6. Ein schwarzes Fenster öffnet sich direkt im richtigen Ordner!

#### Methode B (Klassisch):
1. Drücke die **Windows-Taste**
2. Tippe: `cmd`
3. Öffne die **Eingabeaufforderung**
4. Tippe: `cd C:\Users\DEINNAME\Desktop\qdash`
5. Drücke **Enter**

### 4.2 Prüfe, ob du im richtigen Ordner bist
1. Im schwarzen Fenster, tippe: `dir`
2. Drücke **Enter**
3. Du solltest sehen:
   - Cargo.toml
   - metatron-qso-rs
   - metatron_backend
   - docs
   - usw.

**✅ Du bist im richtigen Ordner!**

### 4.3 Baue das Projekt

1. Im schwarzen Fenster, tippe: `cargo build --workspace --release`
2. Drücke **Enter**
3. **JETZT WIRD ES SPANNEND!** 🎉
   - Du siehst viele grüne und weiße Texte
   - "Compiling..."
   - "Downloading..."
   - "Finished..."
4. **Das dauert 10-20 Minuten beim ersten Mal!**
   - Der Computer baut jetzt die ganze Software
   - Der Lüfter wird wahrscheinlich laut (normal!)
   - **Nicht abbrechen!** Einfach warten ⏳

### 4.4 Warte, bis diese Zeile erscheint:
```
Finished release [optimized] target(s) in XXm XXs
```

**✅ Das Projekt ist gebaut!**

---

## 🎮 Schritt 5: Starte das Dashboard

### 5.1 Starte den Telemetrie-Server

1. Im gleichen schwarzen Fenster, tippe:
   ```
   cargo run --bin metatron_telemetry --release
   ```
2. Drücke **Enter**
3. Warte 5-10 Sekunden
4. Du solltest sehen:
   ```
   INFO metatron_telemetry: Starting Q⊗DASH Telemetry Server
   INFO metatron_telemetry: Listening on http://0.0.0.0:8080
   ```

**✅ Der Server läuft!**

### 5.2 Öffne das Dashboard

1. Öffne deinen **Internet-Browser** (Chrome, Firefox, Edge)
2. Tippe in die Adressleiste: `http://localhost:8080`
3. Drücke **Enter**
4. **🎉 GESCHAFFT!** Das Q⊗DASH Dashboard sollte jetzt erscheinen!

---

## 🎨 Was du jetzt sehen solltest

Das Dashboard zeigt:

### Oben Links: System Status
- **Algorithm**: VQE (der aktuelle Algorithmus)
- **Mode**: Explore (der Modus)
- **ψ (Quality)**: Eine Zahl zwischen 0 und 1
- **ρ (Stability)**: Eine Zahl zwischen 0 und 1
- **ω (Efficiency)**: Eine Zahl zwischen 0 und 1
- **Backend Health**: Drei grüne Punkte ● (SCS, dioniceOS, Q⊗DASH)
- **Quantum Backend**:
  - SIMULATOR badge
  - local_sim
  - 13 qubits

### Oben Rechts: Recent Jobs
- Zeigt die letzten Jobs (am Anfang leer)

### Unten Links: Metrics History
- Ein buntes Diagramm mit drei Linien (ψ, ρ, ω)

### Unten Rechts: Control Actions
- **Start Calibration** Button zum Starten
- **Refresh All** Button zum Aktualisieren

---

## 🎯 Probiere es aus!

### Starte eine Kalibrierung:
1. Klicke auf den **"Start Calibration"** Button (mit dem ▶ Symbol)
2. Eine Nachricht erscheint: "Calibration job started"
3. Im "Recent Jobs" Bereich erscheint ein neuer Job
4. Das Diagramm aktualisiert sich alle 5 Sekunden automatisch

### Zum Beenden:
1. Gehe zurück zum **schwarzen Fenster** (Kommandozeile)
2. Drücke **Strg + C** (Ctrl und C gleichzeitig)
3. Der Server stoppt
4. Das Dashboard funktioniert nicht mehr (normal!)

### Zum erneuten Starten:
1. Im schwarzen Fenster, tippe wieder:
   ```
   cargo run --bin metatron_telemetry --release
   ```
2. Gehe zu `http://localhost:8080` im Browser

---

## 🐛 Fehlerbehebung - Wenn etwas nicht funktioniert

### Problem 1: "rustc ist nicht als interner oder externer Befehl erkannt"
**Lösung:**
1. Schließe die Kommandozeile
2. Öffne sie neu
3. Versuche es nochmal
4. Falls es immer noch nicht geht:
   - Starte den Computer neu
   - Versuche es nochmal

### Problem 2: "error: linker 'link.exe' not found"
**Lösung:**
- Die Visual Studio Build Tools sind nicht richtig installiert
- Gehe zurück zu Schritt 3.2
- Installiere sie nochmal
- **Wichtig**: Wähle "Desktop-Entwicklung mit C++" ✓

### Problem 3: "No such file or directory"
**Lösung:**
- Du bist im falschen Ordner
- Prüfe mit `dir`, ob du Cargo.toml siehst
- Falls nicht, gehe zu Schritt 4.1 zurück

### Problem 4: Das Dashboard zeigt nur "Cannot connect"
**Lösung:**
- Der Server läuft nicht
- Gehe zurück zu Schritt 5.1
- Starte den Server

### Problem 5: Port 8080 ist schon belegt
**Lösung:**
1. Tippe stattdessen:
   ```
   set TELEMETRY_PORT=8081
   cargo run --bin metatron_telemetry --release
   ```
2. Öffne im Browser: `http://localhost:8081`

---

## 📚 Was bedeuten die Sachen?

### Cargo
- Das ist der "Paketmanager" von Rust
- Wie ein Koch, der weiß, welche Zutaten er braucht

### Build/Compile
- Der Computer übersetzt den Code in ein Programm
- Wie ein Übersetzer, der ein Buch in eine andere Sprache übersetzt

### Release
- Eine optimierte Version (schneller, aber dauert länger zum Bauen)

### localhost:8080
- `localhost` = Dein Computer
- `8080` = Die "Tür" (Port), wo das Programm läuft

### Backend
- Der Teil, der die Berechnungen macht (du siehst ihn nicht direkt)

### Frontend/Dashboard
- Der Teil mit den hübschen Grafiken (das, was du im Browser siehst)

---

## 🎓 Fortgeschritten: Backend-Konfiguration (Optional!)

### Wenn du mit IBM Quantum experimentieren willst:

#### 1. Erstelle eine Konfigurations-Datei
1. Öffne Notepad (Windows-Taste, tippe "notepad")
2. Kopiere den Inhalt aus `.env.example` (im qdash Ordner)
3. Speichere als: `C:\Users\DEINNAME\Desktop\qdash\.env`
   - **Wichtig**: Bei "Dateityp" wähle "Alle Dateien (*.*)"
   - Sonst wird es `.env.txt` und funktioniert nicht!

#### 2. Bearbeite die Datei
1. Öffne `.env` mit Notepad
2. Ändere diese Zeilen:
   ```
   IBM_BACKEND_MODE=dry-run
   ```
3. Speichern

#### 3. Was macht das?
- `dry-run` = Testen ohne echte Quantum-Computer
- Keine Kosten, keine API-Token nötig
- Perfekt zum Lernen!

---

## 🎉 Gratulation!

Du hast es geschafft! Du hast:
- ✅ Ein ZIP-File entpackt
- ✅ Rust installiert
- ✅ Build Tools installiert
- ✅ Ein Projekt gebaut
- ✅ Einen Server gestartet
- ✅ Ein Dashboard geöffnet

**Du bist jetzt offiziell ein Q⊗DASH Benutzer!** 🚀

---

## 📞 Hilfe bekommen

Falls du Hilfe brauchst:
1. Lies die Fehlerbehebung oben nochmal
2. Schau in die Datei: `docs/backend_system.md` (auf Englisch)
3. Frage einen Erwachsenen, der sich mit Computern auskennt
4. Öffne ein "Issue" auf GitHub (mit Hilfe eines Erwachsenen)

---

## 🌟 Viel Spaß mit Q⊗DASH!

**Tipp**: Das Dashboard aktualisiert sich automatisch alle 5 Sekunden. Lass es einfach laufen und beobachte, wie sich die Zahlen ändern!

**Sicherheits-Tipp**: Die Standard-Konfiguration ist sicher. Es werden keine echten Quantum-Computer verwendet, nur Simulationen auf deinem Computer. Keine Sorge! 😊
