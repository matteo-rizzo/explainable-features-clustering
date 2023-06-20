import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap

"""
Applies dimensionality reduction via UMAP, performs clustering with HDBSCAN, and then applies UMAP again to reduce the 
dimensionality of the resulting clusters to two dimensions for plotting.

We first apply UMAP to reduce the dimensionality of the data to 10 dimensions. Then, we use HDBSCAN to cluster the
resulting embeddings. Finally, we apply UMAP again to reduce the dimensionality of the resulting clusters to two 
dimensions, which can be easily plotted using matplotlib's scatter() function. The resulting plot shows the different 
clusters, with each cluster assigned a different color.

TODO:
    - Try hierarchical clustering (e.g., random forest)
    - Visualize features (keep in mind that if you want to average the extracted descriptors those should be
        account for SIFT's scale and rotation
    - Use simpler features for MNIST (e.g., simply 7x7 image patch) and try SIFT on more articulated data
    - Check SIFT's parameters here: https://amroamroamro.github.io/mexopencv/matlab/cv.SIFT.detectAndCompute.html
    - Try out:
        * HDBscan: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
        * Dimensionality reduction: https://umap-learn.readthedocs.io/en/latest/
"""


def main():
    # Load your data here, this is just an example
    X = np.random.rand(100, 50)

    # Apply UMAP for dimensionality reduction
    umap_reducer = umap.UMAP(n_components=10, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(X)

    # Perform clustering with HDBSCAN
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    hdbscan_labels = hdbscan_clusterer.fit_predict(umap_embeddings)

    # Apply UMAP again to reduce the dimensionality of the resulting clusters to two dimensions
    umap_reducer_2 = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings_2 = umap_reducer_2.fit_transform(umap_embeddings)

    # Plot the resulting clusters
    plt.scatter(umap_embeddings_2[:, 0], umap_embeddings_2[:, 1], c=hdbscan_labels, s=50, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    main()
