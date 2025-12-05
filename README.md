# AI-Shiteru
## IF3170 Artificial Intelligence - Tugas Besar 2

Repository untuk Tugas Besar 2 mata kuliah IF3170 Artificial Intelligence - Prediksi Status Mahasiswa menggunakan Machine Learning.

---

## Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi status mahasiswa (Dropout, Enrolled, atau Graduate) berdasarkan berbagai fitur akademik dan demografis. Kami mengimplementasikan beberapa algoritma machine learning, terutama **Logistic Regression** dengan pendekatan hierarchical classification.

### Fitur Utama:
- **Data Preprocessing**: Feature engineering, scaling, encoding, dan handling imbalanced data
- **Model Logistic Regression**: Implementasi custom dan sklearn
- **Hierarchical Classification**: Stage 1 (Dropout vs Rest) → Stage 2 (Graduate vs Enrolled)
- **Cross-Validation**: StratifiedKFold untuk evaluasi model yang robust
- **Visualisasi**: Confusion matrix, training animation, dan loss contour plots
- **Model Persistence**: Menyimpan model sebagai `.pkl` untuk deployment

---

## Anggota Kelompok

**Kelompok 06**

| NIM | Nama | Kontribusi |
|-----|------|------------|
| 13523024 | Richard Christian | Support Vector Machine (SVM) implementation, laporan |
| 13523040 | Kenneth Poenadi | Support Vector Machine (SVM) implementation, laporan |
| 13523046 | Ivan Wirawan | Logistic Regression training, laporan |
| 13523086 | Bob Kunanda | Logistic Regression training, laporan |
| 13523104 | M Zahran Ramadhan | Decision Tree Learning (DTL) implementation, laporan |

---

## Struktur Repository

```
AI-Shiteru/
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   └── sample_submission.csv  # Sample submission format
├── src/
│   ├── notebook.ipynb         # Main notebook (comprehensive)
│   ├── notebooklogreg.ipynb   # Logistic Regression focused notebook
│   ├── compare.py             # CSV comparison tool
│   ├── error_analysis.py      # Error analysis utilities
│   └── svm.py                 # SVM implementation
├── submission/
│   └── submission_*.csv       # Generated submission files
├── models/
│   └── model_*.pkl            # Saved models
├── doc/
│   └── report.pdf             # Project report
├── README.md
└── LICENSE
```

---

## Setup dan Instalasi

### Prerequisites
- Python 3.11+
- pip atau conda

### 1. Clone Repository
```bash
git clone https://github.com/KennethhPoenadi/AI-Shiteru.git
cd AI-Shiteru
```

### 2. Buat Virtual Environment (Recommended)
```bash
# Menggunakan venv
python -m venv myenv
source myenv/bin/activate  # macOS/Linux
# atau
myenv\Scripts\activate     # Windows

# Menggunakan conda
conda create -n ai-shiteru python=3.11
conda activate ai-shiteru
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Untuk visualisasi training (optional):**
```bash
# Install FFmpeg untuk video generation
brew install ffmpeg  # macOS
# sudo apt install ffmpeg  # Linux
```

---

## Cara Menjalankan Program

Tersedia 3 notebook berbeda sesuai algoritma yang digunakan:

### Pilih Salah Satu Notebook:

**1. Logistic Regression** (Recommended)
```bash
cd src
jupyter notebook notebooklogreg.ipynb
```
Notebook ini berisi implementasi Logistic Regression dengan hierarchical classification, training animation, dan evaluasi lengkap.

**2. Decision Tree Learning (DTL)**
```bash
cd src
jupyter notebook DTLFeatureHierarchical.ipynb
```
Notebook ini berisi implementasi Decision Tree dengan feature hierarchical.

**3. Support Vector Machine (SVM)**
```bash
cd src
jupyter notebook notebooksvm.ipynb
```
Notebook ini berisi implementasi SVM untuk klasifikasi multiclass.



### Compare Submissions (Python Script):
```bash
cd src
python compare.py
```

Script ini akan membandingkan file referensi dengan submission terbaru di folder `../submission`.

---


## Tools & Libraries

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: ML algorithms, preprocessing, evaluation
- **imbalanced-learn**: SMOTE oversampling
- **matplotlib & seaborn**: Visualization
- **FFmpeg**: Video generation untuk training animation

---

## Fitur Visualisasi

### 1. Confusion Matrix
Setiap fold dari cross-validation menampilkan confusion matrix untuk analisis detail.

### 2. Training Animation
Generate video MP4 yang menampilkan:
- Kontur loss function (log-loss)
- Lintasan parameter θ₀ dan θ₁ selama training
- Permukaan loss 3D
- Grafik loss vs iterasi

```python
model, weights, biases, losses = create_training_animation(
    X, y, lr=0.1, n_iters=300,
    filename='logreg_training_kontur_loss.mp4', fps=15
)
```



**Kelompok 06 - IF3170 Artificial Intelligence**