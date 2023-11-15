from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, SGDRegressor, RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures


import pandas as pd

data = pd.read_csv("data.csv")

features = data.drop(["quality", "Id"], axis=1)
target = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Update the models dictionary
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net': ElasticNet(),
    'Huber Regressor': HuberRegressor(),
    'SGD Regressor': SGDRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'MLP': MLPRegressor(),
    'Naive Bayes': GaussianNB(),  # Note: Naive Bayes is not a regression model, but included for comparison
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'RANSAC Regressor': RANSACRegressor(),
    'Gaussian Process': GaussianProcessRegressor(),
    'Quadratic Regression': LinearRegression()  # Use PolynomialFeatures to introduce quadratic features
}

results = {'Model': [], 'Mean Squared Error': []}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    results['Model'].append(model_name)
    results['Mean Squared Error'].append(mse)

results_df = pd.DataFrame(results)
print(results_df)

param_dist = {
    'n_estimators': np.arange(10, 200, 10),
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': np.arange(2, 20, 2),
    'min_samples_leaf': np.arange(1, 20, 2),
    'bootstrap': [True, False]
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter settings that are sampled
    scoring='neg_mean_squared_error',  # Use mean squared error as the scoring metric
    cv=5,  # Number of cross-validation folds
    verbose=2,
    n_jobs=-1  # Use all available CPU cores
)

# Fit the RandomizedSearchCV to the data
random_search.fit(X_train, y_train)

# Print the best parameters and the corresponding model
print("Best Parameters:", random_search.best_params_)
best_rf_model = random_search.best_estimator_

# Evaluate the model on the test set
test_predictions = best_rf_model.predict(X_test)
mse = mean_squared_error(y_test, test_predictions)
print("Mean Squared Error on Test Set:", mse)
