# 🍅 Smart Tomato Disease Detection System

An AI-powered Computer Vision system that identifies tomato leaf diseases from
photos, giving farmers a fast, automated first-pass diagnosis so treatment can
start before a disease spreads.

---

## 1. Project Overview

**Business problem:** disease diagnosis on farms currently relies on manual
inspection. Farmers may misidentify diseases, agricultural experts aren't
always available, and by the time a disease is caught, it may have already
spread. This system gives an instant, photo-based diagnosis.

**What the system does, per uploaded leaf image:**
- Predicts the most likely disease (or "Healthy")
- Returns a confidence score
- Returns the top-3 most likely diseases (useful when the top prediction is uncertain)
- Flags overall status: **Healthy** or **Diseased**

**Disease classes:**
Tomato Mosaic Virus, Target Spot, Bacterial Spot, Tomato Yellow Leaf Curl
Virus, Late Blight, Leaf Mold, Early Blight, Spider Mites (Two-Spotted Spider
Mite), Septoria Leaf Spot, Healthy Tomato Leaf.

> ⚠️ **Disclaimer:** this is a diagnostic aid built for a class project. It
> should support, not replace, expert agricultural advice for high-value crops.

---

## 2. Dataset Description

- **Expected layout:** one folder per class, images inside:
  ```
  data/train/
      Healthy/*.jpg
      Late_Blight/*.jpg
      Early_Blight/*.jpg
      Bacterial_Spot/*.jpg
      ... (one folder per disease class)
  ```
- This matches the standard PlantVillage-style tomato leaf dataset layout.
- The dataset .zip (~178 MB) was too large to upload directly in this
  delivery — **extract it locally and place the class folders inside
  `data/train/` before running training.** The code only needs the folder
  structure above; exact class-folder names don't need to match the list
  exactly (the training script reads whatever folder names it finds).
- Basic stats to report after running on your full dataset (fill in):
  total images, images per class, image resolution range, train/validation
  split size. `src/preprocessing.py`'s `load_datasets()` prints the class
  names and image counts it found when you run training.

---

## 3. System Architecture (Part 1)

### Input
Farmer/user uploads a **tomato leaf photo** (JPG/PNG) via the API or the
optional Streamlit UI.

### Main Components
```
┌──────────────┐      ┌────────────────────┐      ┌─────────────────────┐
│  Client       │      │  FastAPI service     │      │  CV Model             │
│ (Streamlit /  │─────▶│  (src/api.py)         │─────▶│ (models/*.keras,      │
│  curl / app)  │      │  - auth (API key)     │      │  MobileNetV2-based)   │
└──────────────┘      │  - image decode/resize│      └─────────────────────┘
                        │  - disease-status logic│
                        └──────────┬─────────────┘
                                    │
                                    ▼
                        JSON response:
                        {predicted_disease, confidence,
                         disease_status, top_3}
```

- **Preprocessing module** (`src/preprocessing.py`) — resizes images to
  224×224, builds `tf.data` pipelines with caching/prefetching for training,
  and decodes a single uploaded image for inference. Shared logic keeps
  training and inference consistent.
- **Model layer** — **transfer learning** on top of **MobileNetV2**
  (pretrained on ImageNet): the frozen backbone extracts general visual
  features, and a small custom head (GlobalAveragePooling → Dense →
  Dropout → Dense-softmax) is trained on the tomato leaf classes. A second
  fine-tuning phase unfreezes the last ~30 backbone layers at a low learning
  rate for extra accuracy.
- **API layer** (`src/api.py`) — FastAPI service exposing `POST /predict`
  (single image) behind a required `X-API-Key` header.
- **UI layer** (`app/streamlit_app.py`) — optional demo front-end for
  uploading a photo and viewing the prediction + top-3 breakdown.

### Data Flow
1. User uploads an image → API receives raw bytes (`UploadFile`)
2. Image is decoded and resized to 224×224 (`preprocess_single_image`)
3. Image is passed through the CNN (MobileNetV2 backbone + custom head)
4. Model outputs a probability per class (softmax)
5. API picks the top class, computes `disease_status`, and returns top-3
6. JSON response returned to the client

### Where AI Models Are Used
Only inside the `/predict` endpoint of the API — that's the sole place
inference happens. Everything else (auth, validation, status labeling) is
deterministic code around the model.

### Role of the API
Single integration point between any client (farmer's mobile app, Streamlit
demo, another backend) and the CV model — decouples model retraining/versioning
from the rest of the system, and lets the model be swapped/upgraded (e.g. to
EfficientNet) without changing how clients call it.

### Tech Stack
| Layer | Tool |
|---|---|
| Language | Python 3.10–3.12 (TensorFlow does not yet support 3.13+) |
| Image processing | Pillow, TensorFlow `tf.data` / `image_dataset_from_directory` |
| Model | TensorFlow / Keras — MobileNetV2 transfer learning |
| API | FastAPI + Uvicorn |
| UI (optional) | Streamlit |
| Evaluation | scikit-learn (classification report, confusion matrix) |

---

## 4. AI System Lifecycle (Part 2)

**Data collection:** photos of tomato leaves (healthy and diseased),
organized into one folder per disease class. In production this would come
from farmer photo submissions plus curated/labeled agricultural datasets.

**Role of AI models:** turn a raw leaf photo into a structured diagnosis
(disease name + confidence) — replacing the need for an on-site expert for
a fast first opinion.

**Models used:** one CNN, built via **transfer learning** on **MobileNetV2**
(chosen for being lightweight/fast — good for eventual mobile/edge use on a
farm — while still being highly accurate on image classification thanks to
ImageNet pretraining). The architecture supports swapping in ResNet50 /
EfficientNet / VGG16 / DenseNet / InceptionV3 with minimal code changes
(same `build_model()` pattern in `src/train_model.py`).

**How the model is trained (2 phases):**
1. **Head training:** backbone frozen, only the new classification head
   trains — fast, avoids destroying the pretrained features early on.
2. **Fine-tuning:** the last ~30 backbone layers are unfrozen and trained at
   a very low learning rate (1e-5) to adapt the pretrained features to
   tomato leaves specifically.

Data augmentation (random flip, rotation, zoom, contrast) is applied during
training only, to reduce overfitting on a limited dataset.

**How predictions are used inside the system:** the API's `/predict`
endpoint returns the top disease + confidence + top-3 alternatives; a
`disease_status` (Healthy/Diseased) is derived so a farming app can, for
example, show a green checkmark for healthy leaves and a red alert with
suggested next steps for diseased ones.

**How model performance is evaluated:** accuracy and a full
per-class `classification_report` (precision/recall/F1) plus a confusion
matrix on a held-out validation split (20% of the data), saved to
`outputs/training_results.json` after every training run.

---

## 5. Project Structure
```
tomato_disease_project/
├── data/
│   └── train/                    # place your class-folders here (not included - see §2)
├── src/
│   ├── preprocessing.py          # image loading, augmentation, single-image decode
│   ├── train_model.py            # MobileNetV2 transfer-learning training script
│   └── api.py                    # FastAPI inference service
├── app/
│   └── streamlit_app.py          # optional demo UI
├── models/                       # trained model + class_names.json saved here
├── outputs/                      # evaluation results (JSON) saved here
├── requirements.txt
└── README.md
```

---

## 6. How to Run

### 6.1 Setup
> **Important:** TensorFlow requires **Python 3.10–3.12**. If your machine
> has a newer Python (e.g. 3.13/3.14), either install Python 3.11 alongside
> it, or run this project in **Google Colab** (free, has TensorFlow and a
> free GPU pre-installed — recommended for training).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 6.2 Add your data
Unzip your dataset and place the class folders inside `data/train/` (see
§2 for the expected layout).

### 6.3 Train the model
```bash
python src/train_model.py --data "data/train" --epochs 15 --fine_tune_epochs 5
```
Saves `models/tomato_disease_model.keras`, `models/class_names.json`, and
`outputs/training_results.json` (accuracy, classification report, confusion
matrix).

### 6.4 Run the API
```bash
uvicorn src.api:app --reload --port 8000
```
Docs at `http://localhost:8000/docs`. All `/predict` calls need an
`X-API-Key` header (default demo key: `demo-key-2026` — change via the
`TOMATO_API_KEY` environment variable before real deployment).

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: demo-key-2026" \
  -F "file=@leaf_photo.jpg"
```
**Example response:**
```json
{
  "request_id": "78d21954-...",
  "predicted_disease": "Late_Blight",
  "confidence": 0.91,
  "disease_status": "Diseased",
  "top_3": [
    {"disease": "Late_Blight", "confidence": 0.91},
    {"disease": "Early_Blight", "confidence": 0.05},
    {"disease": "Healthy", "confidence": 0.02}
  ],
  "processed_at": "2026-08-12T19:30:55Z"
}
```

### 6.5 Run the Streamlit demo (optional)
```bash
streamlit run app/streamlit_app.py
```

---

## 7. Evaluation Results

After training on your full dataset, report here (numbers auto-saved to
`outputs/training_results.json`):

| Metric | Value |
|---|---|
| Validation Accuracy | _fill in_ |
| Validation Loss | _fill in_ |

Include the per-class precision/recall/F1 table and a confusion matrix
plot/screenshot, and a short discussion of which diseases are hardest to
tell apart (commonly diseases with similar leaf-spot patterns, e.g. `Early
Blight` vs `Septoria Leaf Spot` vs `Target Spot`).

---

## 8. API Usage Reference

| Endpoint | Method | Body | Auth | Description |
|---|---|---|---|---|
| `/` | GET | — | No | Service info |
| `/health` | GET | — | No | Model load status |
| `/predict` | POST | multipart form: `file=<image>` | Yes (`X-API-Key`) | Single image prediction |

---

## 9. Future Improvements
- Add treatment recommendations per detected disease (bonus feature from the brief)
- Try a stronger backbone (EfficientNetB0/B3) and compare accuracy vs. speed
- Add Grad-CAM visualization to show *which part of the leaf* drove the prediction
- Mobile-friendly UI for in-field use
- Periodic retraining pipeline as new labeled farmer photos come in
