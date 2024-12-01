# import necessary libraries
import os
import cv2
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def preprocess_data(path: str, n_samples: int):
    '''
    This function will load a random sample of data from the folders
    input: 
        path: folder path
        n_samples: number of images to randomly select (default: 200)
    output: list of images and their labels
    '''
    folder = path.split('/')[-2]
    all_files = os.listdir(path)
    np.random.seed(42) # reproducibility purposes
    selected_files = np.random.choice(all_files, min(n_samples, len(all_files)), replace=False) # Randomly select n_samples files
    data = []
    labels = []
    for file in selected_files:
        img = cv2.imread(os.path.join(path, file)) # read image
        img = cv2.resize(img, (256, 256)) # resize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray scale
        data.append(img)
        labels.append(folder.split('_')[0])
    # flatten data and pass to a pipeline for preprocessing as well as one hot encoding
    data = np.array(data).reshape(len(data), -1) # flatten data
    image_pipeline = Pipeline([
        ('scaler', StandardScaler()) # normalize the data
    ])
    # apply the transformation to the data and get the labels ready for one hot encoding
    data_processed = image_pipeline.fit_transform(data).reshape(-1, 256, 256, 1) # reshape data for convolution layers
    return data_processed, labels