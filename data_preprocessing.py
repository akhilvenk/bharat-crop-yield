def load_data(file_path):
    import pandas as pd
    # Load data from a specified file path
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    # Handle missing values
    data = data.dropna()  # Simple example, customize as needed
    return data


def preprocess_data(data):
    # Normalize or scale features if necessary
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


# Example usage:
# data = load_data('path_to_file.csv')
# cleaned_data = clean_data(data)
# preprocessed_data = preprocess_data(cleaned_data)