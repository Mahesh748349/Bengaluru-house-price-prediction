# Bengaluru Housing Price Prediction

A machine learning project for predicting residential property prices in Bengaluru, India. This project combines data cleaning, feature engineering, and model comparison to deliver accurate price predictions through both a Python pipeline and an interactive Streamlit web interface.

## 🎯 Features

- **Data Pipeline**: Clean and transform raw housing data with intelligent feature engineering
- **Model Comparison**: Evaluate Ridge, Random Forest, Extra Trees, and XGBoost models
- **Web Interface**: Interactive Streamlit app for real-time price predictions
- **Production Ready**: Saved models and reproducible pipeline for deployment
- **Accuracy Insights**: Transparent error analysis by property price segment

## 📊 Project Structure

```
housing-project/
├── data/
│   └── train.csv                  # Training dataset (1000 properties)
├── models/
│   └── best_model.joblib          # Trained Ridge regression model
├── outputs/
│   ├── data_summary.csv           # Dataset statistics
│   ├── model_comparison.csv       # Model performance metrics
│   └── run_metadata.json          # Pipeline metadata
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration paths and parameters
│   ├── preprocessing.py           # Data cleaning functions
│   ├── feature_engineering.py     # Feature creation logic
│   ├── modeling.py                # Model training and evaluation
│   ├── utils.py                   # Utility functions
│   └── run_pipeline.py            # Main training pipeline
├── app.py                         # Streamlit prediction interface
├── main.ipynb                     # Jupyter notebook for EDA & experiments
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── ACCURACY_ANALYSIS.md           # Detailed accuracy metrics & limitations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Setup

1. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training pipeline**

   ```bash
   python -m src.run_pipeline
   ```

   This will:
   - Clean and preprocess the data
   - Engineer features
   - Evaluate all candidate models
   - Train the best model
   - Save outputs to `models/` and `outputs/`

## 🌐 Using the Web Interface

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Input Parameters

- **Area (sqft)**: Property size (100-20,000)
- **Location**: Neighborhood name
- **BHK**: Bedroom count (1-10)
- **Bathrooms**: Number of bathrooms (1-10)
- **Balcony**: Number of balconies (0-5)
- **Parking**: Parking spaces (0-5)
- **Furnishing**: Unfurnished, Semi-Furnished, or Fully-Furnished
- **Property Type**: Apartment, Independent House, Villa, or Other
- **Age**: Years since construction (0-100)

### Output

- **Point Estimate**: Base price prediction
- **Confidence Range**: ±20-45% based on price segment

## 📈 Model Performance

| Model         | RMSLE     |
| ------------- | --------- |
| **Ridge**     | 0.2788 ✅ |
| Random Forest | 0.2846    |
| Extra Trees   | 0.2870    |
| XGBoost       | 0.3059    |

**Accuracy by Price Segment:**

| Price Range     | Avg Error  | Best For     |
| --------------- | ---------- | ------------ |
| < 50L           | 43.68%     | Budget homes |
| 50L - 1Cr       | 30.11%     | Mid-market   |
| **1Cr - 1.5Cr** | **19.77%** | **✅ Best**  |
| 1.5Cr - 2.5Cr   | 18.78%     | Premium      |
| > 2.5Cr         | 27.61%     | Luxury       |

See [ACCURACY_ANALYSIS.md](ACCURACY_ANALYSIS.md) for details.

## 📓 Jupyter Notebook

Open `main.ipynb` for:

- Exploratory Data Analysis (EDA)
- Feature distributions
- Locality insights
- Model comparison
- Interactive experimentation

## 🔧 Pipeline

**`src/preprocessing.py`**

- Data cleaning and standardization
- Missing value handling
- Outlier removal (83 records)

**`src/feature_engineering.py`**

- Area-based features (sqft_per_bhk, efficiency metrics)
- Location features (price per sqft, locality statistics)
- Property amenity flags

**`src/modeling.py`**

- Model evaluation via 5-fold cross-validation
- Pipeline with preprocessing + model
- Prediction with confidence ranges

**`app.py`**

- Streamlit web interface
- Real-time price predictions
- Input validation

## 📊 Dataset

- **Size**: 1000 properties, 917 after cleaning
- **Target**: Price in INR (log-transformed for modeling)
- **Features**: 25+ engineered features from 10 raw columns
- **Encoding**: One-hot encoding for categorical variables

## 💡 Use Cases

✅ **Good for:**

- Market screening and property valuation
- Negotiation baseline
- Price range estimation for 1-2.5Cr properties

❌ **Not for:**

- Final investment decisions without expert review
- Ultra-budget/ultra-luxury properties (limited training data)

## 🚀 Future Improvements

1. **Add location intelligence**: Coordinates, distance metrics, amenity data
2. **Expand dataset**: More high-end and budget property samples
3. **Enhance features**: Market trends, construction quality, social indicators
4. **Segment models**: Separate models for different price tiers
5. **API deployment**: FastAPI/Flask for production use

## 📜 License

Open for educational and portfolio use.

---

**Updated**: May 13, 2026 | **Model**: Ridge (RMSLE 0.2788) | **Data**: 917 properties
