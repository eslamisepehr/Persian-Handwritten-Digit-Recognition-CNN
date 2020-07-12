import cv2
import numpy as np
from scipy import io

def load_hoda(training_sample_size=1000, test_sample_size=200, size=5):
    dataset = io.loadmat('./dataset/HodaDataset.mat')
    
    x_train = np.squeeze(dataset['Data'][:training_sample_size])
    y_train = np.squeeze(dataset['labels'][:training_sample_size])
    x_test = np.squeeze(dataset['Data'][training_sample_size:training_sample_size + test_sample_size])
    y_test = np.squeeze(dataset['labels'][training_sample_size:training_sample_size + test_sample_size])
    
    x_train = [cv2.resize(img, dsize=(size, size)) for img in x_train]
    x_test = [cv2.resize(img, dsize=(size, size)) for img in x_test]
    
    x_train = [x.reshape(size * size) for x in x_train]
    x_test = [x.reshape(size * size) for x in x_test]
    
    return x_train, y_train, x_test, y_test