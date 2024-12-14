# Vignette-cnn

## Vignette on implementing distribution-based clustering using cell type data; created as a class project for PSTAT197A in Fall 2023.

<p align="center">
  <img src="img/cnn-workflow.png" alt="CNN Workflow">
</p>

## Contributors

Reese Karo, Daniel Ledvin, Casey Linden, Navin Lo, Will Mahnke

## Table of Contents
- [Vignette Abstract](#vignette-abstract)
- [Repository Content](#repository-content)
- [Contributors](#contributors)
- [Reference List](#reference-list)

## Vignette Abstract:
Our Vignette explains the concepts behind Convolution Neural Networks (CNN) and how we can utilize the computing power to predict image classification. Convolution neural nets are more advanced than regular neural nets or perceptrons, because they have the ability to capture more information from the picture due to the convolution process. Multiple filters (gaussian smoothing, Sobel, Prewitt, Laplacian, etc.) are applied to capture edges, sharp contrasts, and more features that are in images. After applying these filters, the model can pick up on distinct patterns and features, which are then used to make predictions about the content of the image. This hierarchical feature extraction allows CNNs to achieve high accuracy in tasks such as image classification, object detection, and more, making them a powerful tool in the field of computer vision.


## Repository content

 - `data` contains multiple folders of raw data and our processed data used for CNNs:
    
    - `glioma_tumor` contains 901 jpg files of head x-rays performed on patients with a glioma tumor

    - `meningioma_tumor` contains 913 jpg files of head x-rays performed on patients with a meningioma tumor

    - `pituitary_tumor` contains 844 jpg files x-rays performed on patients with a pituitary tumor

    - `no_tumor` contains 438 jpg files of x-rays performed on patients with no tumors present

- `scripts` contains starter **python/jupyter** scripts

    - `Preprocessing.py` contains a function to load images from a subdirectory in `data` and process the photos into a numpy array which will be fed into the pipeline which splits and encodes data and labels.

    - `Modeling.ipynb` contains different models to test out performance and to see how one can improve a model by trying different techniques.

    - `models` folder contains saved models that were ran in the `Modeling.ipynb` notebook 

- `vignette.ipynb` contains the final python notebook for the vignette

- `vignette.html` contains the html render for `vignette.ipynb`

## Reference List
- CNN in Computer Vision
    - https://medium.com/@AIandInsights/convolutional-neural-networks-cnns-in-computer-vision-10573d0f5b00
- Hyperparameter Tuning:
    - https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f
- Tumor CNN Classification
    - https://medicine.washu.edu/news/brain-cancer-vaccine-effective-in-some-patients/
    - https://www.nature.com/articles/s41598-024-65714-w
    - https://www.nature.com/articles/s41598-023-41407-8
- Keras API for ML Classification
    - https://keras.io/api/applications/
- Keras API for Tuning Hyperparameters
    - https://keras.io/keras_tuner/
