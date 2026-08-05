# AutoForecast Lab

> **Production-ready Machine Learning Forecasting Platform**

AutoForecast Lab is an end-to-end machine learning application that transforms raw time-series data into validated datasets, engineered features, trained forecasting models, interactive visualizations, and downloadable forecast results.

The project demonstrates how a forecasting model can be developed, evaluated, and deployed using modern software engineering practices including automated testing, static type checking, continuous integration, containerization, REST APIs, and an interactive web dashboard.

---

## Demo

> **Live Demo:** *(Coming Soon)*

> **Dashboard Screenshot:** *(Coming Soon)*

The application allows users to:

- Upload a CSV time-series dataset
- Automatically validate the dataset
- Generate forecasting features
- Train a machine learning model
- Evaluate forecast performance
- Visualize predictions
- Download forecast results

---

# Key Features

### Data Processing

- Load CSV and Parquet datasets
- Validate missing values and duplicate timestamps
- Detect time-series frequency automatically
- Generate lag, rolling, and calendar features

### Machine Learning

- Chronological train-test splitting
- Expanding-window validation
- Random Forest forecasting model
- Automatic model evaluation
- RMSE and MAE metrics
- Prediction and forecast generation

### Engineering

- Modular forecasting pipeline
- FastAPI REST API
- Interactive Streamlit dashboard
- Docker support
- GitHub Actions CI
- Ruff formatting and linting
- mypy static type checking
- pytest unit testing

---

# Architecture

```text
                  Raw Time-Series Data
                           │
                           ▼
                    Data Loading
                           │
                           ▼
                    Data Validation
                           │
                           ▼
                 Frequency Detection
                           │
                           ▼
                 Feature Engineering
      • Lag Features
      • Rolling Statistics
      • Calendar Features
                           │
                           ▼
             Training Data Preparation
                           │
                           ▼
      Chronological Train/Test Split
        + Walk-Forward Validation
                           │
                           ▼
                 Model Training
                           │
                           ▼
              Model Evaluation
                           │
                           ▼
              Forecast Generation
                           │
                           ▼
        Interactive Dashboard / API
```

---

# Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Data | pandas |
| Machine Learning | scikit-learn |
| Web API | FastAPI |
| Dashboard | Streamlit |
| Testing | pytest |
| Type Checking | mypy |
| Linting | Ruff |
| Environment | uv |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

# Project Structure

```text
forecasting-platform/
│
├── dashboard/                 # Streamlit dashboard
├── data/                      # Example datasets
├── models/                    # Trained models
├── scripts/
├── src/
│   └── forecasting_platform/
│       ├── api/
│       ├── forecasting/
│       ├── training/
│       ├── data_loader.py
│       ├── features.py
│       ├── frequency.py
│       ├── pipeline.py
│       └── validation.py
│
├── tests/
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

# Quick Start

Clone the repository

```bash
git clone https://github.com/<your-username>/forecasting-platform.git

cd forecasting-platform
```

Install dependencies

```bash
uv sync
```

Run the Streamlit dashboard

```bash
uv run streamlit run dashboard/app.py
```

Run the FastAPI service

```bash
uv run fastapi dev src/forecasting_platform/api/main.py
```

Run tests

```bash
uv run pytest
```

---

# Engineering Highlights

This project emphasizes production-oriented machine learning engineering rather than notebook-based experimentation.

Highlights include:

- End-to-end forecasting pipeline
- Modular package architecture
- REST API and interactive dashboard
- Automated feature engineering
- Chronological model evaluation
- Reproducible development environment
- Static type checking
- Automated testing
- Continuous integration

---

# Roadmap

## Completed

- End-to-end forecasting pipeline
- Feature engineering
- Time-series validation
- Random Forest forecasting
- Forecast evaluation
- FastAPI REST API
- Streamlit dashboard
- Docker support
- CI/CD pipeline

## Planned

- Multiple forecasting models
- Hyperparameter optimization
- Model explainability
- Cloud deployment
- Forecast comparison dashboard

---