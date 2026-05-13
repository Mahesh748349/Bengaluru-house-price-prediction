# Bengaluru Housing Price Prediction

Machine learning project for predicting residential property prices in Bengaluru, India. It includes a reproducible training pipeline, saved model artifacts, model comparison outputs, an exploratory notebook, and a Streamlit prediction app.

## Project Structure

```text
housing-project/
|-- app.py                         # Streamlit prediction interface
|-- main.ipynb                     # EDA and experimentation notebook
|-- requirements.txt               # Python dependencies
|-- data/
|   `-- train.csv                  # Training dataset
|-- models/
|   `-- best_model.joblib          # Generated trained model
|-- outputs/
|   |-- data_summary.csv           # Generated dataset summary
|   |-- model_comparison.csv       # Generated CV scores
|   `-- run_metadata.json          # Generated run metadata
`-- src/
    |-- config.py                  # Paths and project settings
    |-- preprocessing.py           # Data cleaning
    |-- feature_engineering.py     # Feature creation
    |-- modeling.py                # Model training and prediction
    |-- utils.py                   # Shared helpers
    `-- run_pipeline.py            # Main training pipeline
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train The Model

```bash
python -m src.run_pipeline
```

The pipeline reads `data/train.csv`, cleans the data, creates engineered features, compares candidate models with 5-fold cross-validation, saves the best model to `models/best_model.joblib`, and writes run outputs to `outputs/`.

Current best model from the saved run:

| Model | RMSLE |
| --- | ---: |
| Ridge | 0.2788 |
| Random Forest | 0.2846 |
| Extra Trees | 0.2870 |
| XGBoost | 0.3059 |

## Run The App

```bash
streamlit run app.py
```

The app expects:

- `data/train.csv`
- `models/best_model.joblib`

If the model file is missing, run the training pipeline first.

## Notebook

Open `main.ipynb` for exploratory data analysis, feature inspection, and modeling experiments.

## Accuracy Notes

The model is best used as a market screening tool, not a final valuation engine. Historical errors are lowest around mid-market and premium properties, while budget and ultra-luxury segments have higher uncertainty because those examples are sparse in the dataset. See `ACCURACY_ANALYSIS.md` for details.
