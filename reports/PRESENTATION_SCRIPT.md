# DataVerse Presentation Script

## 1. Problem

Our track is Bengaluru house price prediction. The task is to predict residential property prices from property features while minimizing RMSLE.

## 2. Data Understanding

The dataset contains property attributes such as area, location, BHK, bathrooms, balcony, parking, furnishing, property type, age, and price.

## 3. Cleaning

We standardized locations, converted area to numeric sqft, checked missing values, and removed unrealistic outliers such as impossible BHK/bathroom counts or abnormal sqft-per-BHK values.

## 4. EDA Insights

Price is skewed, so log transformation helps. Location is a major signal. Area matters, but premium locations can cause large price differences even for similar-sized homes.

## 5. Feature Engineering

We added sqft per BHK, bath per BHK, balcony and parking flags, furnishing flags, property type flags, premium location keyword flags, and location-level median price-per-sqft features.

## 6. Modeling

We compared Ridge Regression, Random Forest, Extra Trees, and XGBoost with 5-fold cross-validation. Ridge Regression currently gives the best RMSLE, so we use it as the saved final model.

## 7. Limitations

The model does not know exact coordinates, metro distance, road access, school quality, or market trends. These would be useful pivot/helper features if provided.

## 8. Impact

The app gives a quick market estimate and confidence range, useful for screening properties and negotiation support.
