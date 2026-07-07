# Butterfly Species Classification

This project classifies butterfly species from images using a YOLOv8 image classification model. The notebook prepares the dataset, trains a classification model, evaluates its performance, and makes predictions on test images.

## Project Overview

The goal of this project is to build an image classification model that can identify different butterfly species. The dataset contains butterfly images and labels stored in CSV files. The images are reorganized into a YOLO-compatible classification folder structure before training.

## Dataset

The dataset used in the notebook contains:

- `Training_set.csv` — training image filenames and labels
- `Testing_set.csv` — test image filenames
- `train/` — labeled training images
- `test/` — unlabeled test images

From the notebook:

- Training images: **6,499**
- Test images: **2,786**
- Number of butterfly classes: **75**

The training data is split into:

- **Train set:** 85%
- **Validation set:** 15%

After splitting, the YOLO dataset contains:

- **5,524 training images**
- **975 validation images**

## Model

The project uses the Ultralytics YOLOv8 classification model:

```python
model = YOLO("yolov8n-cls.pt")
```

Training settings:

```python
results = model.train(
    data=yolo_dataset,
    epochs=30,
    imgsz=224,
    batch=32,
    patience=10,
    project="/content/drive/MyDrive/butterfly_yolo_runs",
    name="butterfly_cls"
)
```

## Results

The trained model was evaluated on the validation set.

| Metric | Score |
|---|---:|
| Top-1 Accuracy | 0.9241 |
| Top-5 Accuracy | 0.9846 |
| Macro F1 Score | 0.9241 |
| Weighted F1 Score | 0.9232 |

## Evaluation

The notebook includes additional evaluation steps:

- Classification report
- Per-class F1 score visualization
- Confusion matrix
- Most confused class pairs

Some of the most confused class pairs were:

| True Class | Predicted Class | Count |
|---|---|---:|
| SOUTHERN DOGFACE | CLOUDED SULPHUR | 4 |
| COPPER TAIL | PURPLISH COPPER | 3 |
| EASTERN DAPPLE WHITE | BECKERS WHITE | 2 |
| PURPLE HAIRSTREAK | GREY HAIRSTREAK | 2 |
| QUESTION MARK | EASTERN COMA | 2 |

## Technologies Used

- Python
- Google Colab
- Google Drive
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Ultralytics YOLOv8

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/dunver486/Butterfly-Specie-Classification.git
cd Butterfly-Specie-Classification
```

2. Install the required library:

```bash
pip install ultralytics
```

3. Open the notebook in Google Colab.

4. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

5. Make sure the dataset zip file is placed in your Google Drive.

In the notebook, the dataset path is:

```python
zip_path = "/content/drive/MyDrive/archive (12).zip"
```

6. Run the notebook cells in order.

## Prediction Example

The notebook predicts classes for images in the test folder:

```python
test_dir = os.path.join(base, "test")
results = model.predict(source=test_dir, save=True)

for r in results[:5]:
    print(r.path, "->", r.names[r.probs.top1], f"({r.probs.top1conf:.2f})")
```

## Project Files

```text
Butterfly-Specie-Classification/
│
├── Butterfly.ipynb
├── README.md
└── images / results / model files
```

## Notes

This project was developed in Google Colab. The model checkpoints and result plots are saved to Google Drive during training and evaluation.

The repository name uses `Specie`, but the correct English term is usually `Species`.
