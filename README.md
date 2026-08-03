# Task 2 AI - Kurt Mikhael Purba - 13524065

Repository ini berisi Proof of Concept Local Search CampusFlow dan eksperimen Decision Tree Learning, Logistic Regression, serta Support Vector Machine dari scratch.

## Struktur

```text
src/
├── local_search/       # PoC Local Search CampusFlow
└── dtl_lr_svm/         # Implementasi DTL, LR, dan SVM
notebooks/
├── local_search/       # Notebook eksperimen Local Search
└── dtl_lr_svm/         # Notebook eksperimen DTL, LR, dan SVM
docs/
├── Task2_AI_Kurt_Mikhael_Purba_13524065.pdf
├── Spesifikasi_Task2_AI_Kurt_Mikhael_Purba_13524065.tex
└── Writeup_Task2_AI_13524065.tex
```

## Menjalankan Local Search

```bash
python src/local_search/main.py
```

POC Local Search mencakup Basic (Steepest-Ascent), Sideways Move, Stochastic, dan Random Restart Hill-Climbing. Output juga menampilkan visualisasi teks perubahan objective dan state antar-iterasi.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Menggunakan Implementasi ML

```python
from src.dtl_lr_svm import (
    CARTClassifierScratch,
    LogisticRegressionScratch,
    LinearSVMScratch,
)
```

Implementasi ML menggunakan NumPy. Preprocessor tabular tersedia melalui `ScratchTabularPreprocessor` dan `ScratchNonlinearPreprocessor`.

## Dokumen

PDF gabungan spesifikasi dan write-up tersedia di `docs/Task2_AI_Kurt_Mikhael_Purba_13524065.pdf`.
