# DL_ImageClassification

Deep Learning Exam 1 (Amir Jafari, Fall 2026): classify leaf photos into six classes (`class1` … `class6`) using **only a multi-layer perceptron** (fully connected layers) trained with TensorFlow. Random guessing gives about 16.7% accuracy.

The final model is an ensemble of 15 MLPs trained on colour and texture features. On the local "test" split (2,592 images) it scores **(macro F1 + Cohen's κ)/2 = 0.908**, with an accuracy of 0.923.

## Results

The local test score is (F1 + κ)/2, the leaderboard formula, computed on the "test" rows of the Excel file. These rows were never used to train or choose a model.

| Day | Main change | Local test |
| --- | --- | --- |
| 1 (10/24) | Baseline MLP on 200 × 200 raw pixels (120,000 inputs) | 0.391 |
| 2 (10/25) | Same network; summary file fixed to match the model | 0.392 |
| 3 (10/26) | Adam, BatchNorm, Kaiming initialization, gradient clipping, deeper and narrower network | 0.420 |
| 4 (10/27) | Geometric data augmentation (flips, rotation, zoom, shift) | 0.473 |
| 5 (10/28) | Early stopping (NND2 Fig. 13.4), learning-rate schedule, class weights, flip test-time augmentation | 0.497 |
| 6 (10/29) | **Feature MLP**: colour and texture histograms instead of pixels | 0.847 |
| 7 (10/30) | 5-fold ensembles, feature augmentation, test-time crops, spot and metadata features | **0.908** |

The largest gain came from changing the input, not the network. A Dense layer has one weight per pixel position, so it cannot recognize a lesion wherever it appears. Histograms of colour and texture count a lesion the same wherever it is, which gives the MLP the position invariance the rules forbid getting from convolution.

## Final model

1. Each JPEG is decoded at 1/8 size (500 × 334). Training images also get 4 cached random views (crops and small lighting changes); test images get 5 views (the whole image and 4 corner crops of 80%).
2. Each view is turned into a feature vector: hue × saturation, brightness, colour moments, excess red/green, local binary patterns at 3 scales, edge strength, centre crop, CIE Lab histograms, tile max-pooling, colour spread, a spot detector, and image metadata from the file header.
3. Each network is z-score → Dense 1024 → 512 → 256 (BatchNorm, ReLU, Dropout 0.4) → Dense 6 softmax. It is trained with Adam, mixup, label smoothing and balanced class weights, and early-stopped on the validation score.
4. 3 feature sets × 5 folds = 15 networks (`model_Matthew1.keras` … `model_Matthew15.keras`). The prediction is the argmax of the mean log-probability over all networks and views.

The full reasoning, figures and references are in `Exam_report/report_Matthew.pdf`.

## Folder structure

```
DL_ImageClassification/
├── Final_Scripts/      final train_Matthew.py, test_Matthew.py and the 15 model files
├── All_Submissions/    day1 … day7 (+ day7_planb): the scripts and models submitted each day
├── Exam_report/        report_Matthew.pdf (and the .docx source)
├── Code/               working folder: scripts, run logs, grid and experiment CSVs, archive/
├── Data/               the images (not in version control)
├── excel/              the Excel file with image ids, targets and the train/test split
└── Exam1_v8.pdf        exam instructions
```

## How to run

Requirements: Python 3, TensorFlow, pandas, numpy, scikit-learn and openpyxl. `test_Matthew.py` installs openpyxl itself if it is missing.

Predict with the submitted models (run from the folder that holds the `.keras` files):

```bash
python3 test_Matthew.py --path /path/to/DL_ImageClassification --split test
```

This writes `results_Matthew.xlsx` (columns `id`, `target`, `split`, `results`) and `summary_Matthew.txt`.

Train from scratch (the experiment settings are at the top of the script):

```bash
python3 train_Matthew.py
```

The first run computes the features of all 14,685 images and caches them in `features_cache/`; later runs start training in seconds. The script trains the grid, builds the 5-fold ensemble, and saves the model files, `grid_Matthew.csv`, `experiments_Matthew.csv` and `early_stopping_Matthew.png`.

## Credits

- Base code: `train_example.py` and `test_to_do.py`, provided with the exam.
- Textbooks: Hagan, Demuth, Beale & De Jesús, *Neural Network Design* (2nd ed., 2014); Hagan & Jafari, *Neural Network Design: Deep Learning* (draft, 2026).
- Papers and tools are listed in the References section of the report.
- Claude (Anthropic) was used as an assistant for code review, writing parts of the scripts, and drafting the report.
