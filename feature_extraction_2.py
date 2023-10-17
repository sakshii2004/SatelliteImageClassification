import cv2
#import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from sklearn.decomposition import PCA
from PIL import Image

pca = pickle.load(open('pca_final.pkl', 'rb'))


def lbp_features_func(images):

  lbp_features = []
  for image in images:
      lbp_image = feature.local_binary_pattern(image, P=8, R=1, method='uniform')

      # Calculate LBP histogram as a texture feature
      hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 9))
      hist = hist.astype("float")
      hist /= (hist.sum() + 1e-7)

      lbp_features.append(hist)
  return lbp_features

def grayscale_features(images):
  gray_features = []  

  for gray_image in images:
      hist, _ = np.histogram(gray_image.ravel(), bins=np.arange(0, 256), range=(0, 255))
      hist = hist.astype("float")
      hist /= (hist.sum() + 1e-7)

      gray_features.append(hist)

  return gray_features



def sobel_edge_detection(images, target_size=(128, 128)):

  sobel_edges = []

  for image in images:

      # Apply Sobel operators to calculate gradients in horizontal and vertical directions
      sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
      sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

      # Calculate gradient magnitude
      gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

      # Resize the gradient magnitude array to the target size
      gradient_magnitude = cv2.resize(gradient_magnitude, target_size)

      # Flatten the gradient magnitude array and append to the list
      sobel_edges.append(gradient_magnitude.flatten())

  return sobel_edges


def red_channel_histogram(images, bins=255):

  red_hist_features = []  

  for image in images:

      # Extract the red channel
      red_channel = image[:, :, 0]

      # Calculate the red channel histogram
      hist_red = cv2.calcHist([red_channel], [0], None, [bins], [0, 256])

      # Normalize the histogram
      hist_red /= (hist_red.sum() + 1e-7)

      # Append the red channel histogram to the feature list
      red_hist_features.append(hist_red.flatten())

  return red_hist_features




def green_channel_histogram(images, bins=255):

  green_hist_features = [] 

  for image in images:
      # Extract the green channel
      green_channel = image[:, :, 1]

      # Calculate the green channel histogram
      hist_green = cv2.calcHist([green_channel], [0], None, [bins], [0, 256])

      # Normalize the histogram
      hist_green /= (hist_green.sum() + 1e-7)

      # Append the green channel histogram to the feature list
      green_hist_features.append(hist_green.flatten())

  return green_hist_features




def blue_channel_histogram(images, bins=255):

  blue_hist_features = []  

  for image in images:
      # Extract the blue channel
      blue_channel = image[:, :, 2]

      # Calculate the blue channel histogram
      hist_blue = cv2.calcHist([blue_channel], [0], None, [bins], [0, 256])

      # Normalize the histogram
      hist_blue /= (hist_blue.sum() + 1e-7)

      # Append the blue channel histogram to the feature list
      blue_hist_features.append(hist_blue.flatten())

  return blue_hist_features




def extract_features(rgb_images, gray_images): 
  #rbg_images and #gray_images are a list of images
  lbp_features = lbp_features_func(gray_images)
  gray_features = grayscale_features(gray_images)
  sobel_edges = sobel_edge_detection(gray_images)
  sobel_edges_pca = pca.transform(sobel_edges)

  red = red_channel_histogram(rgb_images)
  green = green_channel_histogram(rgb_images)
  blue = blue_channel_histogram(rgb_images)

  features = []

  for n in range(len(lbp_features)):

    all_features = np.concatenate((lbp_features[n], gray_features[n], sobel_edges_pca[n], red[n], green[n], blue[n]))
    features.append(all_features)    

  return features

