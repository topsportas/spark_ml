# SparkML MLflow End-to-End Example [Databricks Community Edition]

This project demonstrates an end-to-end machine learning workflow using **Apache Spark**, **MLflow**, and **Databricks** — without requiring AWS or Terraform infrastructure.

---

## Overview

The notebook trains a wine quality classification model using the UCI Wine Quality dataset, performs hyperparameter tuning with XGBoost and Hyperopt, registers the best model in MLflow Model Registry, and runs batch inference using Spark Delta tables.

### What was done

- Imported the `mlflow-end-to-end-example` notebook into Databricks using the Databricks CLI
- Adapted the notebook to run on **Databricks Community Edition** (serverless compute, Unity Catalog enabled) without AWS or Terraform
- Fixed all Unity Catalog compatibility issues (model stages → aliases, DBFS paths → managed Delta tables)
- Successfully ran the full notebook end-to-end

---

## Adaptations Made

The original notebook was designed for a Databricks workspace provisioned on AWS via Terraform. The following changes were made to run it on Databricks Community Edition:

| Issue | Original | Fix Applied |
|---|---|---|
| MLflow not available | Standard runtime | `%pip install --upgrade typing_extensions mlflow` + `dbutils.library.restartPython()` |
| `SparkTrials` not supported on serverless | `SparkTrials(parallelism=10)` | `Trials()` |
| Model stages not supported in Unity Catalog | `transition_model_version_stage(..., stage="Production")` | `set_registered_model_alias(..., "production", version)` |
| Model loading by stage not supported | `models:/wine_quality/production` | `models:/wine_quality@production` |
| Public DBFS root disabled | `dbfs:/<username>/delta/wine_data` | Managed Delta table via `saveAsTable("wine_data")` |
| Model serving requires Premium tier | Databricks Model Serving endpoint | Local model prediction comparison |

---

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) installed and configured
- A Databricks workspace (Community Edition is sufficient)

### Configure Databricks CLI

```bash
pip install databricks-cli
databricks configure --token
# Host: https://community.cloud.databricks.com
# Token: <your personal access token>
```

---

## Import Notebook to Databricks

```bash
for notebook in notebooks/*.py; do
    name=$(basename "$notebook" .py)
    databricks workspace import /Shared/$name --file "$notebook" --language PYTHON --format SOURCE --overwrite
done
```

---

## Notebook Walkthrough

### 1. Install Dependencies

```python
%pip install --upgrade typing_extensions mlflow
dbutils.library.restartPython()
```

### 2. Load Data

Reads the UCI Wine Quality dataset (white and red wines) from `/databricks-datasets/` — pre-loaded in all Databricks workspaces.

```python
white_wine = pd.read_csv("/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/databricks-datasets/wine-quality/winequality-red.csv", sep=";")
```

### 3. Train Baseline Model (Random Forest)

Trains a `RandomForestClassifier` with MLflow tracking. Logs AUC metric and registers the model in the Unity Catalog Model Registry with a `production` alias.

```python
with mlflow.start_run(run_name='untuned_random_forest'):
    model = RandomForestClassifier(n_estimators=10, ...)
    ...
    mlflow.pyfunc.log_model("random_forest_model", ...)

client.set_registered_model_alias(model_name, "production", model_version.version)
```

### 4. Hyperparameter Tuning (XGBoost + Hyperopt)

Runs 96 trials of XGBoost hyperparameter search using Hyperopt. Each trial is tracked as a nested MLflow run.

```python
trials = Trials()
with mlflow.start_run(run_name='xgboost_models'):
    best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=96, trials=trials)
```

The best model (highest AUC) is promoted to `production` in the Model Registry.

### 5. Batch Inference with Spark Delta

Saves training data as a managed Delta table and applies the production model for batch predictions.

```python
spark.sql("DROP TABLE IF EXISTS wine_data")
spark_df.write.format("delta").mode("overwrite").saveAsTable("wine_data")

new_data = spark.read.table("wine_data")
new_data = new_data.withColumn("prediction", apply_model_udf(udf_inputs))
```

### 6. Model Serving

The model is registered in Unity Catalog under `workspace.default.MyRegisteredModel` with a full input/output signature.

> **Note:** Live HTTP endpoint serving requires Databricks Premium or Enterprise. The notebook demonstrates equivalent predictions using the model loaded directly from the registry.

```python
model = mlflow.pyfunc.load_model(f"models:/wine_quality@production")
model_evaluations = model.predict(X_test[:5])
```

---

## Results

- Baseline Random Forest AUC: **~0.854**
- Best XGBoost AUC after hyperparameter sweep: **~0.90**
- Model registered in Unity Catalog with alias `production`
- Batch inference successfully run on Delta table
- Model predictions verified against test set

---

## Repository Structure

```
.
├── notebooks/
│   └── mlflow-end-to-end-example.py   # Databricks notebook (source format)
├── terraform/                          # Original AWS/Terraform infrastructure (not used)
└── README.md
```
