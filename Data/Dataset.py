import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, concatenate, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
import gdown
from ipywidgets import interact, IntSlider

IMG_PATH = "reduced/imgs"
MASK_PATH = "reduced/masks"
EPOCHS = 40

!wget "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/Deep%20Dives/Common%20Files/Lung%20Segmentation%20Notebook/reduced.zip"
!unzip -oq reduced.zip
