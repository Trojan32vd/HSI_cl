import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
from spectral.io import envi
from spectral import view_cube
import spectral
default_dpi = mpl.rcParamsDefault['figure.dpi']

def read_hyperspectral_data(radiance_file, header_file):
    """
    Read hyperspectral data from ENVI format files
    """
    # Read the header file
    try:
        header = envi.read_envi_header(header_file)
        
        # Extract wavelength information
        wavelengths = np.array([float(w) for w in header['wavelength']])
        
        # Read the radiance data
        img = envi.open(header_file, radiance_file)
        data = img.load()
        
        # Create RGB visualization
        # Using default bands from header if available, otherwise use middle of visible spectrum
        if 'default bands' in header:
            rgb_bands = [int(b) for b in header['default bands']]
        else:
            # Approximate bands for RGB (around 650nm, 550nm, 450nm)
            rgb_bands = [
                np.argmin(np.abs(wavelengths - 650)),
                np.argmin(np.abs(wavelengths - 550)),
                np.argmin(np.abs(wavelengths - 450))
            ]
        
        # Create RGB image
        rgb_image = np.dstack([data[:,:,b] for b in rgb_bands])
        
        # Normalize RGB image for display
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        
        return data, wavelengths, rgb_image, header
        
    except Exception as e:
        print(f"Error reading hyperspectral data: {str(e)}")
        return None, None, None, None

def display_rgb(rgb_image, title="RGB Composite"):
    """
    Display RGB composite image
    """
    plt.figure(figsize=(10,10))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Main execution
if __name__ == "__main__":
    # File paths
    radiance_file = 'afx102_1_2026_radiance.dat'
    header_file = 'afx102_1_2026_radiance.hdr'
    
    # Read the data
    data, wavelengths, rgb_image, header = read_hyperspectral_data(radiance_file, header_file)
    
    if data is not None:
        # Display RGB composite
        display_rgb(rgb_image, "Hyperspectral RGB Composite")
        
        # Print basic information
        print(f"Data shape: {data.shape}")
        print(f"Wavelength range: {wavelengths.min():.2f}nm - {wavelengths.max():.2f}nm")
        print(f"Number of bands: {len(wavelengths)}")