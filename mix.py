import pandas as pd

def modify_labels(file_path, output_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Convert label column from float to int
    data['label'] = data['label'].astype(int)

    # Save the modified dataset
    data.to_csv(output_path, index=False)

# Usage example
input_file_path = '/Users/chamodyaavishka/Desktop/EMAIL/product/mixed_dataset.csv'  # Replace with the path to your dataset file
output_file_path = '/Users/chamodyaavishka/Desktop/EMAIL/product/newmix.csv'  # Path where the modified dataset will be saved

modify_labels(input_file_path, output_file_path)
