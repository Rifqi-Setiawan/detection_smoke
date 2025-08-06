# Smoke Detection System

Sistem deteksi asap/merokok menggunakan YOLOv8 dengan Flask web API dan real-time camera monitoring.

## 🚀 Fitur

- **Real-time Detection**: Deteksi asap dan rokok secara real-time menggunakan kamera
- **Web API**: Interface REST API untuk monitoring status deteksi
- **High Accuracy**: Menggunakan model YOLOv8 yang telah di-training dengan dataset khusus
- **Confidence Scoring**: Menampilkan tingkat kepercayaan deteksi
- **Detection Counter**: Menghitung jumlah deteksi harian

## 🛠️ Teknologi yang Digunakan

- **Deep Learning**: YOLOv8 (Ultralytics)
- **Backend**: Flask
- **Computer Vision**: OpenCV
- **Frontend Communication**: Flask-CORS
- **Image Processing**: NumPy, PIL

## 📋 Persyaratan Sistem

- Python 3.8+
- Webcam/Camera
- GPU (recommended untuk performa optimal)

## 🔧 Instalasi

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd smokeDetection
   ```

2. **Buat virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Dataset** 📧
   Dataset tidak disertakan dalam repository ini. Untuk mendapatkan dataset:
   - Hubungi developer melalui email atau issue di repository ini
   - Dataset berisi gambar dengan label 'cigarette' dan 'smoking'
   - Format dataset: YOLO format dengan struktur folder train/val/test

5. **Download Pre-trained Model**
   - Model weights tersimpan di `runs/detect/train6/weights/best.pt`
   - Jika tidak tersedia, download dari releases atau hubungi developer

## 🚀 Menjalankan Aplikasi

1. **Pastikan kamera terhubung**

2. **Jalankan aplikasi**
   ```bash
   python smoke_detector.py
   ```

3. **Akses API**
   - Video stream: `http://localhost:5000/video`
   - Status API: `http://localhost:5000/status`

## 📡 API Endpoints

### GET `/video`
Streaming video real-time dengan deteksi bounding box

### GET `/status`
```json
{
  "detections_today": 5,
  "detected": true,
  "confidence": 0.87
}
```

- `detections_today`: Jumlah deteksi hari ini
- `detected`: Status deteksi saat ini (3 detik buffer)
- `confidence`: Tingkat kepercayaan deteksi terakhir

## 📁 Struktur Project

```
smokeDetection/
├── smoke_detector.py      # Main application
├── data.yaml             # Dataset configuration
├── requirements.txt      # Python dependencies
├── README.md            # Documentation
├── .gitignore           # Git ignore rules
├── datasets/            # Dataset folder (not included)
│   └── smoking_combined/
│       ├── train/
│       ├── valid/
│       └── test/
├── runs/                # Training results
│   └── detect/
│       └── train6/
│           └── weights/
│               └── best.pt  # Trained model
└── exp10-new/          # Experiment results
```

## ⚙️ Konfigurasi

### Model Configuration
- **Confidence Threshold**: 0.25 (dapat diubah di `smoke_detector.py`)
- **Detection Cooldown**: 5 detik antara notifikasi
- **Detection Buffer**: 3 detik untuk status "detected"

### Camera Settings
- Default camera index: 0
- Resolusi: Native camera resolution
- FPS: Optimal berdasarkan hardware

## 🔧 Training Model (Opsional)

Jika ingin melakukan training ulang:

1. Pastikan dataset sudah tersedia
2. Install YOLO dependencies
3. Jalankan training:
   ```bash
   yolo train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640
   ```

## 🐛 Troubleshooting

### Camera tidak terdeteksi
- Pastikan camera tidak digunakan aplikasi lain
- Coba ubah camera index di `cv2.VideoCapture(0)` ke angka lain

### Model tidak ditemukan
- Pastikan file `runs/detect/train6/weights/best.pt` ada
- Download model dari releases atau hubungi developer

### Performa lambat
- Gunakan GPU jika tersedia
- Kurangi resolusi atau confidence threshold
- Pastikan webcam mendukung resolusi yang digunakan

## 📞 Kontak & Support

Untuk mendapatkan dataset atau bantuan lebih lanjut:
- 📧 Hubungi developer melalui email
- 🐛 Buat issue di repository ini
- 💬 Diskusi di section Discussions

## 📄 Lisensi

Project ini dibuat untuk tujuan edukasi dan penelitian. Silakan hubungi developer untuk penggunaan komersial.

## 🙏 Acknowledgments

- **Ultralytics** untuk YOLOv8 framework
- **OpenCV** untuk computer vision tools
- **Flask** untuk web framework

---

⚠️ **Catatan**: Dataset tidak disertakan dalam repository ini karena ukurannya yang besar. Silakan hubungi developer untuk mendapatkan akses dataset.
