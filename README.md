# vignette-cnn

This project uses convolution neural networks (CNNs) for image-classificiation and prediction

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
