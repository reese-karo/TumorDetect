# import necessary libraries
import os
import cv2
import numpy as np

def preprocess_data(path: str, n_samples: int, seed: int):
    '''
    This function will load a random sample of data from the folders.
    
    Input: 
        path (str): Folder path.
        n_samples (int): Number of images to randomly select.
    
    Output:
        tuple: (data_processed (np.ndarray), labels (list[str]))
    '''
    all_files = os.listdir(path) # create a list of all files in the path
    np.random.seed(seed=seed)  # reproducibility
    selected_files = np.random.choice(all_files, min(n_samples, len(all_files)), replace=False) # randomly select n_samples files
    data = []
    labels = []
    for file in selected_files: # iterate through the selected files
        try:
            img = cv2.imread(os.path.join(path, file))  # read image
            if img is None:
                continue
            img = cv2.resize(img, (256, 256))  # resize image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            data.append(img)
            labels.append(os.path.basename(path).split('_')[0])  # Use folder name for labels
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Convert to numpy array and normalize pixel values to [0, 1]
    data = np.array(data) / 255.0
    data = data.reshape(len(data), 256, 256, 1)  # Reshape for CNNs
    
    return data, labels