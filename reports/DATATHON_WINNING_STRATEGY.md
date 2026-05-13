# DataVerse Winning Strategy

## What The Contest Requires

The guide and presentation describe a 10-hour datathon with four tracks. For the Bengaluru housing track, the goal is to estimate residential property prices and minimize RMSLE. Judges care about:

- Model rigor: algorithm choice and hyperparameter justification.
- Technical pivot: ability to merge new helper data released at 3 PM.
- Stability: consistent performance on hidden/private test data.
- Interpretability: strong EDA, feature importance, and clear plots.
- Pitch quality: explaining why predictions make sense.

## What This Project Already Covers

- Reproducible training pipeline in `src/run_pipeline.py`.
- Cleaning and outlier handling in `src/preprocessing.py`.
- Feature engineering in `src/feature_engineering.py`.
- Model comparison in `src/modeling.py`.
- Best current model: `ridge`.
- Streamlit demo app in `app.py`.
- EDA and model plots in `reports/figures/`.
- 2 PM audit report in `reports/DATATHON_EDA_REPORT.md`.

## What To Do To Maximize Winning Chances

1. Prepare the 2 PM EDA pitch around cleaning, hidden signals, and plots.
2. Be ready for the 3 PM pivot by writing new helper data as CSV into `data/` and merging it by location or property identifiers.
3. Track before/after metric changes after adding pivot data.
4. Add ablation evidence: compare baseline features versus engineered features.
5. Keep the final GitHub repo clean and freeze it before the deadline.

## Strong Judge Explanation

We used RMSLE because housing prices vary widely, and log error is more stable for price prediction. We tested Ridge, Random Forest, Extra Trees, and XGBoost, then selected the best performer by cross-validation. The main insight is that location premium and engineered space-quality ratios explain more than raw area alone.
