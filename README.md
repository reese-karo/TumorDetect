# vignette-cnn

Our Vignette explains the concepts behind Convolution Neural Networks (CNN) and how we can utilize the computing power to predict image classification. Convolution neural nets are more advanced than regular neural nets or perceptrons, because they have the ability to capture more information from the picture due to the convolution process. Many filters are applied to capture things like edges, sharp contrasts, and more features that are in images. After applying many 

- WIP: include an image of the CNN process. Image -> conv layer with x filters of 3x3 kernel, then max pool, then more layers to capture info, then flatten to be passed into a deep neural network that can be passed into a final output layer with the softmax activation, so that we can get probabilities for each class. then the output with the highest probability is the predicted class.

## Repository content

 - `data` contains multiple folders of raw data and our processed data used for CNNs:
    
    - `glioma_tumor` contains 901 jpg files of head x-rays performed on patients with a glioma tumor

    - `meningioma_tumor` contains 913 jpg files of head x-rays performed on patients with a meningioma tumor

    - `pituitary_tumor` contains 844 jpg files x-rays performed on patients with a pituitary tumor

    - `normal_tumor` contains 438 jpg files of x-rays performed on patients with no tumors present

- WIP: add information for processed folder when it's added

- `scripts` contains some starter py scripts

    - `preprocessing.py` contains a function to pull images from a subdirectory in `data` and process the photos into a numpy array

    - WIP: add description for other py files that get added tp scripts

- `vignette.ipynb` contains the final python notebook for the vignette

- `vignette.html` contains the html render for `vignette.ipynb`

## Reference List
- 