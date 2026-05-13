# Housing Price Prediction Accuracy Analysis

The model has meaningful predictive signal, but Bengaluru housing prices are noisy and strongly location-dependent. Treat predictions as a pricing reference, not as a final valuation.

## Error By Price Segment

| Price range | Average error | Notes |
| --- | ---: | --- |
| < 50L | 43.68% | Few budget examples in the dataset |
| 50L - 1Cr | 30.11% | Mid-market homes |
| 1Cr - 1.5Cr | 19.77% | Strongest segment in this data |
| 1.5Cr - 2.5Cr | 18.78% | Premium segment |
| > 2.5Cr | 27.61% | Only a small luxury sample |

## Main Limitations

- Extreme price bands have limited training examples.
- Important location signals are missing, such as distance to metro stations, IT parks, schools, CBD areas, and major roads.
- Micro-location effects are large in Bengaluru; similar property profiles can differ heavily by neighborhood quality and exact address.
- The current model uses a compact dataset and should be updated as more listings become available.

## Practical Use

Good for:

- Quick market screening
- Negotiation baseline
- Comparing similar property profiles
- Educational or portfolio demonstration

Not recommended for:

- Final purchase decisions
- Investment commitments
- Legal, lending, or appraisal use without expert review

The Streamlit app shows a prediction range to make this uncertainty visible.
