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


def denoiser(images):
  denoised_images = []
  for image in images:
      denoised_image = cv2.medianBlur(image, 3)
      denoised_images.append(denoised_image)

  return denoised_images

def apply_clahe(images, clip_limit=2, tile_grid_size=(8, 8)):
  enhanced_images = []
  for image in images:
      lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

      # Split LAB channels
      l_channel, a_channel, b_channel = cv2.split(lab_image)

      # Apply CLAHE to the L channel (luminance)
      clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
      enhanced_l_channel = clahe.apply(l_channel)
      enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
      enhanced_rgb_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2RGB)
      enhanced_images.append(enhanced_rgb_image)
  return enhanced_images


def unsharp_masking_color(images, alpha=1.5, sigma=1.0):
  enhanced_images = []
  for image in images:
      blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)        
      sharpened_image = cv2.addWeighted(image, alpha + 1.0, blurred_image, -alpha, 0)
      enhanced_images.append(sharpened_image)

  return enhanced_images


def preprocess(RGB_images): #RGB_images is a list
  denoised_images = denoiser(RGB_images)
  contrast_enhanced_images = apply_clahe(denoised_images)
  enhanced_images = unsharp_masking_color(contrast_enhanced_images)

  enhanced_gray = []
  for i in range(len(enhanced_images)):
      gray = cv2.cvtColor(enhanced_images[i],cv2.COLOR_RGB2GRAY)
      enhanced_gray.append(gray)

  return enhanced_images, enhanced_gray

# the enhanced_images is a list of enhanced RGB Images
# the enhanced_gray is a list of enhanced GRAY Images