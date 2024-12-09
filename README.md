# vignette-cnn

Our Vignette explains the concepts behind Convolution Neural Networks (CNN) and how we can utilize the computing power to predict image classification. Convolution neural nets are more advanced than regular neural nets or perceptrons, because they have the ability to capture more information from the picture due to the convolution process. Many filters are applied to capture things like edges, sharp contrasts, and more features that are in images. After applying many filters through multiple activation functions, the CNN's output is a number corresponding to a category of the outcome variable (in this case, the type of cancer or lack there of). 

- WIP: include an image of the CNN process. Image -> conv layer with x filters of 3x3 kernel, then max pool, then more layers to capture info, then flatten to be passed into a deep neural network that can be passed into a final output layer with the softmax activation, so that we can get probabilities for each class. then the output with the highest probability is the predicted class.

## Contributors

Reese Karo, Daniel Ledvin, Casey Linden, Navin Lo, Will Mahnke

## Repository content

 - `data` contains multiple folders of raw data and our processed data used for CNNs:

    - **WIP: include folder for raw images?**
    
    - `glioma_tumor` contains 901 processed images of head x-rays performed on patients with a glioma tumor

    - `meningioma_tumor` contains 913 processed images of head x-rays performed on patients with a meningioma tumor

    - `pituitary_tumor` contains 844 processed images x-rays performed on patients with a pituitary tumor

    - `normal_tumor` contains 438 processed images of x-rays performed on patients with no tumors present

- WIP: add information for processed folder when it's added

- `scripts` contains some starter py scripts

    - `drafts` contains files used in the data preprocessing and our initial CNNs

        - `Modeling.ipynb` contains our initial, simple CNN architecture as well as a function that uses hyperparameter tuning to optimize the CNN architecture

        - `preprocessing.py` contains a function to pull images from a subdirectory in `data` and process the photos into a numpy array 

- `vignette.ipynb` contains the final python notebook for the vignette

- `vignette.html` contains the html render for `vignette.ipynb`

## Reference List
- 
