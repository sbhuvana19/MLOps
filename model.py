import pickle
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression


def train_model():
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    model = LinearRegression()
    model.fit(X, y)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model