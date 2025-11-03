
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from typing import Dict, Any

class RegressionModel:
    """
    Base class for regression models.
    Provides a common interface for training, evaluating, and predicting.
    """
    def __init__(self, model):
        """
        Initializes the RegressionModel with a scikit-learn model.

        Args:
            model: A scikit-learn compatible regression model.
        """
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def split_data(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the data into training and testing sets.

        Args:
            features (pd.DataFrame): The input features.
            target (pd.Series): The target variable.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        print("Data split into training and testing sets.")

    def train(self):
        """
        Trains the regression model.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not been split yet. Call split_data() first.")
        
        print(f"Training {self.model.__class__.__name__}...")
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the trained model on the test set.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics
                              (MSE, MAE, R2, RMSE).
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data has not been split yet. Call split_data() first.")

        print("Evaluating model...")
        predictions = self.model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        rmse = np.sqrt(mse)

        metrics = {
            "Mean Squared Error (MSE)": mse,
            "Mean Absolute Error (MAE)": mae,
            "R-squared (R2)": r2,
            "Root Mean Squared Error (RMSE)": rmse,
        }
        print("Model evaluation complete.")
        return metrics

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on new data.

        Args:
            input_data (pd.DataFrame): The input data for prediction.

        Returns:
            np.ndarray: The predictions.
        """
        print("Making predictions...")
        return self.model.predict(input_data)

class SimpleLinearRegressionModel(RegressionModel):
    """
    A simple linear regression model.
    """
    def __init__(self):
        super().__init__(LinearRegression())

class RandomForestRegressorModel(RegressionModel):
    """
    A Random Forest Regressor model.
    """
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__(RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
