# import necessary libraries
import os
import cv2
import numpy as np
def preprocess_data(path: str, size_output: int):
    '''
    This function will load the data from the given path and preprocess it.
    
    Input: 
        path (str): Folder path.
    Output:
        tuple: (data_processed (np.ndarray), labels (list[str]))
    '''
    all_files = os.listdir(path)  # create a list of all files in the path
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Total files in {path}: {len(image_files)}")  # Debug: print total number of files

    data = []
    labels = []
    for file in image_files:  # iterate through the selected files
        try:
            img = cv2.imread(os.path.join(path, file))  # read image
            if img is None:
                print(f"Warning: {file} could not be read and will be skipped.")  # Debug: warn if image is None
                continue
            img = cv2.resize(img, (size_output, size_output))  # resize image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            img = img/255.0 # normalize pixel values to [0, 1]
            data.append(img)
            labels.append(os.path.basename(path).split('_')[0])  # Use folder name for labels
        except Exception as e:
            print(f"Error processing file {file}: {e}")  # Debug: print exception message
    data = np.array(data)  # Shape: (samples, height, width, channels)
    print(f"Processed {len(data)} images.")  # Debug: print number of processed images
    return data, labels