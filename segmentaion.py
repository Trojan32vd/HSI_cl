import numpy as np
from spectral import envi
import sklearn.preprocessing as prep
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_subset_hyperspectral(reflectance_file, header_file, subset_size=(500, 500)):
    """
    Load a subset of hyperspectral data from ENVI format files
    """
    try:
        # Load the hyperspectral data header first
        img = envi.open(header_file, reflectance_file)
        
        # Get original dimensions
        rows, cols, bands = img.shape
        
        # Calculate subset dimensions
        subset_rows = min(subset_size[0], rows)
        subset_cols = min(subset_size[1], cols)
        
        # Calculate starting position for centered subset
        start_row = (rows - subset_rows) // 2
        start_col = (cols - subset_cols) // 2
        
        print(f"Loading subset of size {subset_rows}x{subset_cols} from {rows}x{cols} image...")
        
        # Load only the subset using memmap
        data = img.open_memmap()
        subset = data[start_row:start_row+subset_rows, 
                     start_col:start_col+subset_cols, :]
        
        # Convert to numpy array in memory
        subset = np.array(subset)
        
        # Reshape to 2D array (pixels x bands)
        pixels = subset_rows * subset_cols
        X = subset.reshape(pixels, bands)
        
        # Remove invalid pixels (zeros or negatives)
        valid_pixels = np.all(X > 0, axis=1)
        X = X[valid_pixels]
        
        # Apply normalization
        X = prep.StandardScaler().fit_transform(X)
        
        return X, (subset_rows, subset_cols), valid_pixels
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def perform_clustering(X, n_clusters=4):
    """
    Perform clustering to segment different seafloor classes
    """
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        return labels, kmeans
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        raise

def reduce_dimensions(X, n_components=10):
    """
    Reduce dimensionality using PCA
    """
    try:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"Explained variance with {n_components} components: {explained_var:.2%}")
        return X_reduced, pca
    except Exception as e:
        print(f"Error in dimension reduction: {str(e)}")
        raise

def reconstruct_image(labels, image_shape, valid_pixels):
    """
    Reconstruct the segmentation map to original image dimensions
    """
    try:
        full_labels = np.zeros(np.prod(image_shape[:2]))
        full_labels[valid_pixels] = labels
        return full_labels.reshape(image_shape[:2])
    except Exception as e:
        print(f"Error reconstructing image: {str(e)}")
        raise

def save_results(segmentation_map, cluster_means, output_prefix="test_subset"):
    """
    Save results to files
    """
    try:
        # Save segmentation map
        np.save(f"{output_prefix}_segmentation.npy", segmentation_map)
        
        # Save cluster means
        np.save(f"{output_prefix}_cluster_means.npy", cluster_means)
        
        # Save plots
        plt.savefig(f"{output_prefix}_results.png")
        print(f"Results saved with prefix: {output_prefix}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        raise

def main(reflectance_file, header_file, subset_size=(500, 500), n_clusters=4):
    try:
        # Load and preprocess subset of data
        print("Loading subset of hyperspectral data...")
        X, image_shape, valid_pixels = load_subset_hyperspectral(
            reflectance_file, 
            header_file, 
            subset_size
        )
        
        print(f"Loaded data shape: {X.shape}")
        
        # Reduce dimensions
        print("Reducing dimensions...")
        X_reduced, pca = reduce_dimensions(X)
        
        # Perform clustering
        print("Clustering pixels...")
        labels, kmeans = perform_clustering(X_reduced, n_clusters)
        
        # Reconstruct segmentation map
        segmentation_map = reconstruct_image(labels, image_shape, valid_pixels)
        
        # Calculate mean spectra for each cluster
        cluster_means = []
        for i in range(n_clusters):
            cluster_pixels = X[labels == i]
            mean_spectrum = np.mean(cluster_pixels, axis=0)
            cluster_means.append(mean_spectrum)
        cluster_means = np.array(cluster_means)
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        # Plot segmentation map
        plt.subplot(121)
        plt.imshow(segmentation_map)
        plt.title('Segmentation Map')
        plt.colorbar(label='Cluster ID')
        
        # Plot mean spectra
        plt.subplot(122)
        for i, mean_spectrum in enumerate(cluster_means):
            plt.plot(mean_spectrum, label=f'Cluster {i}')
        plt.title('Mean Spectra per Cluster')
        plt.xlabel('Band Number')
        plt.ylabel('Normalized Reflectance')
        plt.legend()
        
        plt.tight_layout()
        
        # Save results
        save_results(segmentation_map, cluster_means)
        
        return segmentation_map, cluster_means, labels, kmeans
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Replace with your file paths
        reflectance_file = "afx102_1_2026_reflectance.dat"
        header_file = "afx102_1_2026_radiance.hdr"
        
        # Test with a small subset (500x500 pixels)
        subset_size = (111211, 512)
        
        segmentation_map, cluster_means, labels, kmeans = main(
            reflectance_file,
            header_file,
            subset_size=subset_size
        )
    except Exception as e:
        print(f"Program failed: {str(e)}")

#    reflectance_file = "afx102_1_2026_reflectance.dat"
#    header_file = "afx102_1_2026_radiance.hdr"