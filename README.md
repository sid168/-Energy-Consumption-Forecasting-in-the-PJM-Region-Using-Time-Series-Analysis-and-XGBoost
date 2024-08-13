# Energy-Consumption-Forecasting-in-the-PJM-Region-Using-Time-Series-Analysis-and-XGBoost

---
Develop advanced forecasting model for energy consumption in the PJM region using time series analysis and the XGBoost algorithm. By capturing historical patterns and incorporating various factors, the model aims to provide accurate predictions to optimize energy distribution, enhance grid stability, and support informed decision-making.


---

# Time Series Forecasting with PJM East Data

This project demonstrates time series forecasting using PJM East electrical power consumption data (PJME_MW). The data is analyzed, processed, and modeled to predict future energy consumption using various techniques, including feature engineering and machine learning models.

## Project Overview

The notebook walks through the following key steps:

1. **Data Loading and Exploration**:
   - Load and explore the PJM East Megawatt (PJME_MW) dataset.
   - Conduct Exploratory Data Analysis (EDA) to understand the data distribution, trends, and seasonality.

2. **Feature Engineering**:
   - Convert datetime columns and extract useful features such as `year`, `month`, `day`, `hour`, and other time-based features.
   - Aggregate and analyze data to identify patterns and trends across different time scales.

3. **Data Visualization**:
   - Use libraries like `matplotlib`, `seaborn`, and `plotly` to create visualizations.
   - Visualize the time series data, rolling statistics, and distribution of features.

4. **Train-Test Split**:
   - Implement a train-test split on the time series data to ensure a proper division of training and test sets.
   - Address potential issues such as data leakage.

5. **Modeling**:
   - Use the XGBoost regressor to model and predict the energy consumption.
   - Evaluate model performance using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).

6. **Model Interpretation and Visualization**:
   - Visualize feature importance and inspect the decision trees.
   - Compare actual data with predictions and analyze errors to identify underfitting and other issues.

7. **Prediction Analysis**:
   - Analyze the prediction errors and identify the days with the worst predictions.
   - Discuss the smoothness of predictions and potential underfitting.

## Data

The dataset used in this notebook is the `PJME_hourly.csv`, containing hourly energy consumption data for the PJM East region.

- **Columns**:
  - `Datetime`: Timestamp of the recorded data.
  - `PJME_MW`: Energy consumption in megawatts.

## Requirements

To run the notebook, the following libraries are required:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `xgboost`
- `scikit-learn`

You can install the necessary packages using pip:

```bash
pip install pandas numpy matplotlib seaborn plotly xgboost scikit-learn
```

## Usage

1. Clone the repository and navigate to the project directory.
2. Ensure that the `PJME_hourly.csv` dataset is in the working directory.
3. Open the Jupyter notebook `Time Series Forecasting.ipynb` and run the cells sequentially.
4. Analyze the outputs, visualizations, and model predictions.

## Conclusion

This notebook provides a comprehensive walkthrough of time series forecasting using machine learning techniques. It covers everything from data exploration and feature engineering to model training, evaluation, and interpretation. The approach can be adapted to other time series datasets for similar forecasting tasks.


