Car Price Prediction Model 🚗
A machine learning project to predict the market price of cars based on technical specifications and features using Python and Scikit-Learn.

Workflow Steps

Step 1: Data Exploration (EDA)

1.Loaded the dataset and analyzed numerical distributions using df.describe().
2.Visualized data scales and identified the need for scaling/normalization due to the high range of the price column.
3.Checked for correlations using a heatmap to identify features like enginesize and horsepower as key price drivers.

Step 2: Data PreprocessingFeature Engineering: 

1.Handling Categorical Data: Used One-Hot Encoding (pd.get_dummies) with drop_first=True to convert text labels (like fuel type or body style) into numerical binary values
2.Feature Selection: Separated the dataset into Features ($X$) and Target ($y$).

Step 3: Model Training

1.Train-Test Split: Divided the data (80% training, 20% testing) to ensure the model could be evaluated on "unseen" data.
2.Random Forest Regressor: Implemented an ensemble "Decision Tree" model to capture complex non-linear relationships.

step 4:

1.EvaluationMeasured performance using R-squared (R^2) and Mean Absolute Error (MAE).
2.Achieved an R^2 of 0.92, meaning the model explains 92% of the price variations.
