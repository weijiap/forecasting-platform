# Forecasting Platform

A production-oriented forecasting platform that transforms raw time-series data into validated features, trained models, evaluation metrics, and exportable forecasts.

The project demonstrates an end-to-end machine learning workflow for time-series forecasting, including data validation, automated feature engineering, chronological train-test splitting, model training, evaluation, prediction generation, testing, containerization, and continuous integration.


## Core Features

- Load CSV and Parquet datasets
- Validate missing values and duplicate timestamps
- Detect time-series frequency
- Generate lag, rolling, and calendar features
- Preserve chronological order during train-test splitting
- Support expanding-window validation
- Train a Random Forest regression model
- Evaluate forecasts with RMSE and MAE
- Automatically select the best evaluated model
- Generate prediction and forecast DataFrames
- Run the complete workflow through one forecasting pipeline
- Validate code with Ruff, mypy, pytest, and GitHub Actions
- Build reproducible environments with Docker and uv


## Architecture

```text
Raw Time-Series Data
        |
        v
Data Loader
        |
        v
Data Validation
        |
        v
Frequency Detection
        |
        v
Feature Engineering
  - Lag features
  - Rolling statistics
  - Calendar features
        |
        v
Training Data Preparation
        |
        v
Chronological Split / Walk-Forward Validation
        |
        v
Model Training
        |
        v
Model Evaluation and Selection
        |
        v
Prediction Generation
        |
        v
Forecast DataFrame

## Architecture

```text
Raw Time-Series Data
        |
        v
Data Loader
        |
        v
Data Validation
        |
        v
Frequency Detection
        |
        v
Feature Engineering
  - Lag features
  - Rolling statistics
  - Calendar features
        |
        v
Training Data Preparation
        |
        v
Chronological Split / Walk-Forward Validation
        |
        v
Model Training
        |
        v
Model Evaluation and Selection
        |
        v
Prediction Generation
        |
        v
Forecast DataFrame


## Project Structure

```text
forecasting-platform/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
├── models/
├── scripts/
├── src/
│   └── forecasting_platform/
│       ├── forecasting/
│       │   ├── __init__.py
│       │   └── predict.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── benchmark.py
│       │   ├── dataset.py
│       │   ├── splitter.py
│       │   └── trainer.py
│       ├── data_loader.py
│       ├── features.py
│       ├── frequency.py
│       ├── pipeline.py
│       └── validation.py
├── tests/
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
## Status

Under active development.