# Credit Risk Detection – MLOps and Explainable AI Project
## 1. Project Overview

This project implements an **end-to-end machine learning system** for **credit risk detection**, following modern **MLOps practices** and later extending the system with **Explainable AI (XAI)** techniques.

The goal of the project is not only to train a predictive model, but to build a **reproducible, deployable, and observable ML pipeline**, similar to what is used in real industrial environments. In a second stage, explainability methods will be applied to understand **why the model makes certain decisions**, which is especially important in high-stakes domains such as finance.

The project covers:
- Data versioning and reproducibility
- Model training and experiment tracking
- Containerization and deployment
- Cloud deployment with Kubernetes
- Monitoring with Prometheus
- Explainability and model interpretation (XAI)

## 2. Problem Description

The task is a **binary classification problem** to predict whether a loan applicant is likely to **default on a credit**.

- **Input**: Applicant information (financial status, credit history, personal data, etc.)
- **Output**: Probability of default
- **Why explainability matters**: Credit decisions affect people directly, so it is important to understand and justify model predictions.

## 3. Dataset

The project uses the **German Credit Dataset** from the UCI Machine Learning Repository.

### Dataset characteristics
- Tabular data
- Numerical and categorical features
- Binary target variable (default / no default)

### Data handling
- The dataset is downloaded and processed automatically
- Data is versioned using **DVC**
- This ensures full reproducibility of experiments

## 4. Model

- **Model**: Tree-based classifier (scikit-learn)
- **Reason for choice**:
  - Works well with tabular data
  - Stable and interpretable
  - Compatible with explainability methods such as SHAP and feature importance

The model is trained through a **DVC pipeline**, and all experiments are tracked using **MLflow**.

## 5. MLOps Architecture

The project follows a modular and production-oriented structure:

- **DVC**:  
  Data and model versioning, pipeline reproducibility
- **MLflow**:  
  Experiment tracking and metric logging
- **Docker**:  
  Containerization of services
- **Docker Compose**:  
  Local orchestration of services
- **Kubernetes**:  
  Cloud deployment
- **Prometheus**:  
  Monitoring and metrics collection

### Services
- **API service (Flask)**  
  Serves model predictions
- **Web interface (Streamlit)**  
  User interface for submitting inputs and viewing predictions

## 6. Running the Project Locally (Docker Compose)

### Prerequisites
- Docker
- Docker Compose
- Git

### Steps

1. Clone the repository:
```bash
git clone <repository_url>
cd credit-risk-detection
```

2. Train teh model
```bash
python src/data.py
dvc repro
```

3. Build images and containers
```bash
docker build -t natley30/api -f Dockerfile.api .
docker build -t natley30/web -f Dockerfile.web .
docker-compose up
```

### Notes on DVC and Training

This repository uses **DVC with a remote storage configured on DagsHub**.  
Because of this, users cloning the repository **will not have access to the DVC remote by default**, since it requires personal credentials.

For this reason, the recommended workflow is to **retrain the model locally** after cloning the repository.

This will:
- Download and preprocess the dataset
- Train the model locally using the parameters defined in params.yaml
- Generate a new model.pkl file

Users are encouraged to modify params.yaml to experiment with different model configurations.
DVC will automatically detect parameter changes and re-run the training stage.

### Notes on Docker Images
The Docker image names used in this project (for example natley30/api and natley30/web) correspond to the original author’s Docker Hub account.

If you are running the project under a different Docker Hub account, you should:

1. Build the images using your own namespace, for example:
```bash
docker build -t <your_username>/api -f Dockerfile.api .
docker build -t <your_username>/web -f Dockerfile.web .
```

2. Update the image names accordingly in docker-compose.yaml.

## 7. Cloud Deployment (Kubernetes)
The project can be deployed to a Kubernetes cluster (for example, Google Kubernetes Engine).

### Deployed components
- API Deployment + ClusterIP Service
- Web Deployment + LoadBalancer Service

### Access
- The Web service is exposed via an external IP
- The API service is internal and accessed via Kubernetes DNS

## 8. Monitoring
Both the API and the Web interface expose Prometheus metrics. These metrics can be scraped by Prometheus and visualized using Grafana.

## 9. Explainable AI (XAI)
In the next phase of the project, explainability techniques will be applied to the trained model.

The objectives of the XAI part are:
- Understand which features influence predictions
- Explain individual decisions (local explanations)
- Detect potential bias or unstable behavior
- Support more transparent and responsible decision-making

Planned methods include:
- Feature importance
- SHAP values
- Local explanations for individual predictions
- Sanity checks and model behavior analysis

This makes the project suitable for high-stakes decision scenarios, such as credit approval.

## 10. Project Purpose

This project is intended to demonstrate:
- Practical MLOps skills
- Deployment and monitoring of ML systems
- Responsible use of machine learning through explainability
