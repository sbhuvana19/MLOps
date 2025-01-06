import pickle
from model import train_model


def test_model_training():
    model = train_model()
    assert model is not None
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model is not None
