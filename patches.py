import numpy as np
from skimage.util import view_as_windows
from sklearn.cluster import KMeans

# Load your image
image = ...

# Define the size of your patches
patch_size = (32, 32)

# Extract patches from your image using a sliding window
patches = view_as_windows(image, patch_size, step=16)
patches = patches.reshape(-1, patch_size[0], patch_size[1], 3)

# Load your training images
train_images = [...]

# Extract patches from your training images
train_patches = []
for image in train_images:
    patches = view_as_windows(image, patch_size, step=16)
    patches = patches.reshape(-1, patch_size[0] * patch_size[1] * 3)
    train_patches.append(patches)

train_patches = np.concatenate(train_patches, axis=0)

# Perform k-means clustering to build a vocabulary of visual words
num_clusters = 1000
kmeans = KMeans(n_clusters=num_clusters, n_jobs=-1)
kmeans.fit(train_patches)
visual_words = kmeans.cluster_centers_

from scipy.spatial.distance import cdist

# Compute the distance between each patch and each visual word
distances = cdist(patches.reshape(-1, patch_size[0] * patch_size[1] * 3), visual_words)

# Find the closest visual word to each patch
closest_words = np.argmin(distances, axis=1)

# Compute the histogram of visual word occurrences for each patch
histograms = []
for i in range(len(patches)):
    histogram = np.zeros(num_clusters)
    for j in range(patch_size[0] * patch_size[1]):
        histogram[closest_words[i * patch_size[0] * patch_size[1] + j]] += 1
    histograms.append(histogram)
