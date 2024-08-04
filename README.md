# Cat and Dog Classifier

This repository contains code and resources for a Cat and Dog image classifier using deep learning. The project includes a Jupyter notebook for training and evaluating the model, a pre-trained model file, and a Streamlit application for a user-friendly interface to make predictions.

## Repository Contents

- `Cat_Dog_Classifier.ipynb`: Jupyter notebook containing the code for training, evaluating, and saving the model.
- `cats_vs_dogs_model.h5`: Pre-trained model file in HDF5 format.
- `streamlitui.py`: Streamlit application script for deploying the model with a user interface.
- `testing data`: Directory containing testing data for the model.

## Requirements

To run the code in this repository, you need the following libraries:

- Python 3.x
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib

You can install the required packages using the following command:

```sh
pip install tensorflow keras streamlit numpy pandas matplotlib
```

## Usage

### Jupyter Notebook

To train and evaluate the model, open the `Cat_Dog_Classifier.ipynb` notebook and run the cells in sequence. The notebook includes detailed explanations of each step.

### Streamlit Application

To run the Streamlit application, use the following command:

```sh
streamlit run streamlitui.py
```

This will start a local web server and open the application in your default web browser. You can use the interface to upload images and get predictions on whether the image is of a cat or a dog.

### Testing Data

The `testing data` directory contains images that can be used to test the model. You can add your own images to this directory and use them with the Streamlit application.

## Model Training

The model is trained using a Convolutional Neural Network (CNN) architecture. The training process includes data augmentation, model compilation, and model fitting. The trained model is saved as `cats_vs_dogs_model.h5`.

## Training vs Validation Metrics
Below are the graphs showing the training vs validation accuracy and loss over epochs.

![accuracy](https://github.com/user-attachments/assets/5a9511cb-c3a2-4a14-841e-1c13251e29e1)

![loss](https://github.com/user-attachments/assets/d6984c71-16f7-4283-aaef-a213aa52c4cf)


## Acknowledgements

- The dataset used for training the model is from the [Kaggle Cats and Dogs Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset).
- Special thanks to the developers of TensorFlow, Keras, and Streamlit for providing excellent tools for machine learning and web application development.
