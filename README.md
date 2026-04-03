# Network Intrusion Detection System (NIDS)

A production-ready, **dataset-agnostic** machine learning pipeline for detecting network intrusions in real-time. Supports multiple benchmark datasets (**UNSW-NB15**, **CIC-IDS2018**, **CICIDS2017**), achieves **97% accuracy** with Random Forest, and is served via a FastAPI REST API with Docker support.

## Supported Datasets

| Dataset | Year | Records | Attack Types | Source |
|---------|------|---------|-------------|--------|
| **UNSW-NB15** (default) | 2019 | 2.5M+ | 9 | UNSW Canberra |
| **CIC-IDS2018** | 2018 | 16M+ | 14+ | CIC, Canada |
| **CICIDS2017** | 2017 | 2.8M | 14 | CIC, Canada |

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐
│  Raw CSVs   │───>│ Preprocessing│───>│   Training   │───>│  Model    │
│ (Multi-     │    │  & Feature   │    │ 5 Classifiers│    │ .joblib   │
│  dataset)   │    │  Engineering │    │ + CV + Eval  │    │ artifact  │

└─────────────┘    └──────────────┘    └─────────────┘    └─────┬─────┘
                                                                │
                   ┌──────────────┐    ┌─────────────┐          │
                   │   Client     │<──>│  FastAPI     │<─────────┘
                   │  (Swagger)   │    │  /predict    │
                   └──────────────┘    └─────────────┘
```

## Results

| Model           | Train Acc | Test Acc | CV Mean | Training Time |
|-----------------|-----------|----------|---------|---------------|
| **Random Forest** | **0.9999** | **0.9714** | **0.9680** | **12.4s** |
| Decision Tree   | 0.9999    | 0.9625   | 0.9591  | 1.2s          |
| KNN             | 0.9823    | 0.9543   | 0.9510  | 0.8s          |
| AdaBoost        | 0.5912    | 0.5834   | 0.5780  | 8.6s          |
| SVM             | 0.9701    | 0.9412   | 0.9380  | 45.2s         |

## Project Structure

```
├── src/
│   ├── config.py          # Central configuration (paths, hyperparameters)
│   ├── preprocess.py      # Data loading, cleaning, feature engineering
│   ├── train.py           # Model training, comparison, artifact saving
│   └── evaluate.py        # Metrics visualization and plot generation
├── api/
│   └── app.py             # FastAPI REST API for real-time prediction
├── data/
│   └── download.py        # Kaggle dataset download script
├── models/                # Saved model artifacts (.joblib)
├── plots/                 # Generated evaluation visualizations
├── main.py                # End-to-end pipeline runner
├── Dockerfile             # Container deployment
├── requirements.txt       # Python dependencies
└── README.md
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/RAHUL-26/Network-Intrusion-Detection-System.git
cd Network-Intrusion-Detection-System
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Option A: Kaggle CLI (requires ~/.kaggle/kaggle.json)
python data/download.py                          # Downloads default (UNSW-NB15)
python data/download.py --dataset cicids2018     # Downloads CIC-IDS2018
python data/download.py --dataset cicids2017     # Downloads CICIDS2017

# Option B: Manual download
# UNSW-NB15:  https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
# CIC-IDS2018: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
# CICIDS2017:  https://www.kaggle.com/datasets/cicdataset/cicids2017
# Extract CSVs to data/
```

### 3. Train the Pipeline

```bash
python main.py --data-dir data/ --dataset unsw_nb15     # Default
python main.py --data-dir data/ --dataset cicids2018
python main.py --data-dir data/ --dataset cicids2017
```

This will:
- Load and sample the selected dataset
- Clean data (remove duplicates, NaN, inf, small classes)
- Remove highly correlated features (threshold: 0.85)
- Undersample to balance classes
- Train 5 classifiers (Random Forest, Decision Tree, KNN, SVM, AdaBoost)
- Save best model to `models/nids_model.joblib`
- Generate evaluation plots in `plots/`

### 4. Serve the API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

### 5. Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ..., 0.5]}'
```

Response:
```json
{
  "prediction": "BENIGN",
  "confidence": 0.9714,
  "model": "Random Forest"
}
```

### API Endpoints

| Method | Endpoint    | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | API and model health status          |
| POST   | `/predict`  | Classify a network traffic sample    |
| GET    | `/features` | List expected feature names          |
| GET    | `/docs`     | Interactive Swagger documentation    |

## Docker

```bash
# Build
docker build -t nids-api .

# Run
docker run -p 8000:8000 nids-api
```

## ML Pipeline Details

### Preprocessing
- **Sampling**: 25% stratified sample from each CSV (configurable)
- **Cleaning**: Duplicate removal, NaN/inf handling, class filtering (>10K samples)
- **Feature Engineering**: Correlation-based feature selection (threshold: 0.85)
- **Balancing**: Random undersampling for class balance
- **Normalization**: Z-score normalization + StandardScaler

### Models Trained
- **Random Forest** (max_depth=40) — Best performer
- **Decision Tree** — Fast baseline
- **K-Nearest Neighbors** — Distance-based
- **Support Vector Machine** — Margin-based
- **AdaBoost** — Boosting ensemble

### Evaluation
- Train/test accuracy, confusion matrices, classification reports
- 5-fold cross-validation scores
- Training time benchmarks
- All plots auto-generated in `plots/`

## Datasets

| Dataset | Link | Attack Types |
|---------|------|-------------|
| **UNSW-NB15** | [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms |
| **CIC-IDS2018** | [Kaggle](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv) | DDoS, Brute Force, Bot, Infiltration, Web Attack, DoS, Heartbleed, PortScan |
| **CICIDS2017** | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) | BENIGN, DDoS, PortScan, Bot, Infiltration, Web Attack, Brute Force, Heartbleed |

## Tech Stack

- **Python 3.11+**
- **scikit-learn** — ML models and preprocessing
- **pandas / NumPy** — Data manipulation
- **FastAPI** — REST API serving
- **Docker** — Containerization
- **matplotlib / seaborn** — Visualization

## License

MIT
