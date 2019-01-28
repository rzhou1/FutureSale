# Sale prediction - Time series dataset
The dataset is part of a Coursera course (How to win a data science competition) / Kaggle challenge for predicting future sales. 

Here we provide a CatBoost model for predict retail future sale. The solution includes the following parts:

Part 1. Original data preprocessing.
Part 2. Aggregate daily data by month and define train and test data.
Part 3. Extract and engineer features.
Part 4. Train, val, test data splits.
Part 5. Model (include hyperparameters tuning).

The model performance has been significantly improved by clipping count values to [1%, 99%], extracing lag features, and averaging out predictions from several hold-out datasets for modeling.
