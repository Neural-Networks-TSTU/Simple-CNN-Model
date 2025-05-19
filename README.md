# Simple-CNN-Model

This project presents a minimalistic implementation of a convolutional neural network (CNN) for the task of image classification. The model provides a basic architecture and tools for training and testing.

## Preparing the dataset
To work with the model, you need to prepare the data as follows:
- Create a folder structure where each folder is named after the class name and has corresponding images inside.
- Run the `dataset_spliter.py` script to automatically split the data into training and validation sets.


## How to start learning

To start the training process, do the following:
```bash
python train.py
```
The model will be saved in .pth format for later use.

## Visualization of the training process
During training, a graph of the change in the loss function and accuracy for the training and validation samples is generated:
<div align="center">
  <img src="checkpoints/training_curves_cnn.png" alt="График потерь" width="800"/>
</div>

## Model Checking Tools
### Checking on Single Images
To quickly check a model on a single image, use:

```bash
python predict.py --image "path/to/image.jpg" --model "path/to/model.pth"
```
### Bulk testing
To evaluate the quality of the model on a large dataset:
- Prepare the test dataset:
  - Create a folder with test images
  - Create a labels.csv file with a match between file names and their true labels
- Run the testing script:

```bash
python test_model.py
```

## Dependencies
Install all required dependencies:
- For GPU:
  ```bash
  pip install -r requirements-gpu.txt
  ```
- For CPU:
  ```bash
  pip install -r requirements-cpu.txt
  ```
