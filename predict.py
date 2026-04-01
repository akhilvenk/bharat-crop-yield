import joblib
import numpy as np


def load_model(model_path):
    """Load a trained model from the specified path."""
    model = joblib.load(model_path)
    return model


def predict_yield(model, features):
    """Predict crop yield using the loaded model and provided features."""
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return prediction[0]


if __name__ == '__main__':
    # Example usage
    model_path = 'path_to_your_trained_model.pkl'
    features = [10, 5, 200]  # replace with appropriate values
    model = load_model(model_path)
    yield_prediction = predict_yield(model, features)
    print(f'Predicted crop yield: {yield_prediction}')