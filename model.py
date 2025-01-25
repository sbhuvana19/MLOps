import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder


def train_model():
    # Load CSV file
    data = pd.read_csv("data/iris.csv")

    # Separate features and target
    X = data.iloc[:, :-1]  # All columns except the last one as features
    y = data.iloc[:, -1]  # The last column as the target

    # Encode the target variable if it's categorical
    if y.dtypes == "object":  # Check if the target is a string (categorical)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run():
        # Initialize Ridge regression and define hyperparameter grid
        model = Ridge()
        param_grid = {"alpha": np.logspace(-4, 4, 10)}

        # Perform Grid Search with cross-validation
        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring="r2",
            cv=5
        )
        grid_search.fit(X_train, y_train)

        # Log the best parameters and metrics to MLflow
        mlflow.log_param("model", "Ridge")
        best_alpha = float(
            grid_search.best_params_["alpha"]
        )  # Extract best alpha
        mlflow.log_param("best_alpha", best_alpha)

        mlflow.log_metric("best_r2_score", grid_search.best_score_)

        # Use the best estimator for predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Evaluate and log the test MSE
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("test_mse", mse)
        print(
            "Mean Squared Error on test set: "
            + str(mse)
        )

        # Save the best model to a file
        joblib.dump(best_model, "model.joblib")

        # Display the best parameters and results
        best_params = {
         key: float(value) for key, value in grid_search.best_params_.items()
        }
        print("Best Parameters:", best_params)
        print("Best R2 Score (CV):", grid_search.best_score_)


# Run the training function
if __name__ == "__main__":
    train_model()
