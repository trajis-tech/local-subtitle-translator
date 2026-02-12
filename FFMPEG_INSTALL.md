# FFmpeg 下載與安裝說明

本專案 **Run A（音訊分析）** 與 **Run C/D（視訊擷取）** 需要 ffmpeg。

- **Windows**：`start.bat` 會自動嘗試下載可攜版到 `runtime\ffmpeg`；若自動下載失敗或您希望手動安裝，請依下列方式操作。
- **Linux / macOS**：`start.sh` 僅檢查 PATH 中是否有 ffmpeg，若無則顯示警告；請依系統套件管理員安裝（如 `apt install ffmpeg`、`brew install ffmpeg`），或參閱下方手動安裝方式。

---

## 一、自動安裝（主腳本）

執行 **`start.bat`** 時會：

1. 檢查系統 PATH 是否已有 `ffmpeg`
2. 若無，檢查 `runtime\ffmpeg\bin\ffmpeg.exe` 是否存在
3. 若不存在，從 BtbN 下載 **Windows 64-bit GPL Shared** 並解壓到 `runtime\ffmpeg`
4. 將 `runtime\ffmpeg\bin` 加入本次執行之 PATH

無須手動安裝時，直接執行 `start.bat` 即可。

---

## 二、手動下載與安裝（自動失敗時）

### 2.1 下載

**來源：BtbN FFmpeg Builds（Windows 64-bit）**

| 版本 | 連結 | 說明 |
|------|------|------|
| **GPL Shared（較小，約 89 MB）** | [ffmpeg-master-latest-win64-gpl-shared.zip](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip) | 建議，與主腳本自動下載相同 |
| GPL 完整（約 202 MB） | [ffmpeg-master-latest-win64-gpl.zip](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip) | 靜態連結，較大 |
| 其他版本 | [BtbN Releases](https://github.com/BtbN/FFmpeg-Builds/releases) | 可選 win64-lgpl 等 |

### 2.2 解壓與放置

1. 下載上述 zip 後解壓，會得到一個資料夾（例如 `ffmpeg-master-latest-win64-gpl-shared`）。
2. 將該資料夾**重新命名為** `ffmpeg`，並**整個移動到**專案目錄下：
   ```
   專案根目錄\runtime\ffmpeg\
   ```
3. 確認路徑為：
   ```
   runtime\ffmpeg\bin\ffmpeg.exe
   ```
   若解壓後是 `runtime\ffmpeg\ffmpeg-xxx\bin\ffmpeg.exe`，請改為把 `ffmpeg-xxx` 的內容移到 `runtime\ffmpeg\`，或將 `ffmpeg-xxx` 重新命名為 `ffmpeg` 並放在 `runtime\` 下。

### 2.3 讓主腳本使用

- **方式 A（建議）**：保持上述目錄結構，直接再執行 **`start.bat`**。腳本會偵測 `runtime\ffmpeg\bin\ffmpeg.exe` 並把 `runtime\ffmpeg\bin` 加入 PATH，無須系統安裝。
- **方式 B**：將 `runtime\ffmpeg\bin` 加入系統環境變數 PATH，則本機任何程式都能使用 ffmpeg。

---

## 三、系統安裝（選用）

若希望整機都可使用 ffmpeg，可改為系統安裝：

### 3.1 winget（Windows 11 / 較新 Windows 10）

```bat
winget install --id=Gyan.FFmpeg -e
```

安裝後請重新開啟命令列或重新開機，再執行 `start.bat`。

### 3.2 Chocolatey

```bat
choco install ffmpeg
```

### 3.3 官方 builds

1. 前往 [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. 點 Windows 圖示，選 **Windows builds by BtbN** 或 **gyan.dev**
3. 下載 win64 版本，解壓到例如 `C:\ffmpeg`
4. 將 `C:\ffmpeg\bin` 加入系統環境變數 PATH

---

## 四、驗證

在命令提示字元執行：

```bat
ffmpeg -version
```

若有顯示版本資訊，表示 PATH 設定正確，本專案即可使用 Run A 與視訊相關功能。
