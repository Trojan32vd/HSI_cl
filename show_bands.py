import spectral
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import Normalize

class HyperspectralViewer:
    def __init__(self, data_file, header_file):
        # Load the data
        self.img = spectral.envi.open(header_file, data_file)
        self.current_band = 0
        self.rgb_bands = [46, 28, 9]  # Approximately R:650nm, G:550nm, B:450nm
        
        # Create the main figure
        self.fig = plt.figure(figsize=(15, 8))
        self.setup_layout()
        
    def setup_layout(self):
        # Create grid for subplots
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[4, 1])
        
        # Single band display
        self.ax_band = self.fig.add_subplot(gs[0, 0])
        self.ax_band.set_title(f'Band {self.current_band + 1} - {self.img.bands.centers[self.current_band]:.2f}nm')
        
        # RGB composite
        self.ax_rgb = self.fig.add_subplot(gs[0, 1])
        self.ax_rgb.set_title('RGB Composite')
        
        # Spectral profile
        self.ax_spec = self.fig.add_subplot(gs[1, :])
        self.ax_spec.set_title('Spectral Profile')
        self.ax_spec.set_xlabel('Wavelength (nm)')
        self.ax_spec.set_ylabel('Radiance')
        
        # Add slider for band selection
        self.ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        self.band_slider = Slider(
            self.ax_slider, 'Band', 0, self.img.nbands-1,
            valinit=self.current_band, valstep=1
        )
        self.band_slider.on_changed(self.update_band)
        
        # Display initial images
        self.update_display()
        
    def update_band(self, val):
        self.current_band = int(val)
        self.update_display()
        
    def normalize_band(self, band_data):
        """Normalize band data to 0-1 range"""
        valid_data = band_data[~np.isnan(band_data)]
        vmin, vmax = np.percentile(valid_data, (2, 98))
        normalized = np.clip((band_data - vmin) / (vmax - vmin), 0, 1)
        return normalized
        
    def create_rgb(self):
        """Create RGB composite from three bands"""
        rgb = np.dstack([
            self.normalize_band(self.img.read_band(b))
            for b in self.rgb_bands
        ])
        return rgb
        
    def update_display(self):
        # Clear previous plots
        self.ax_band.clear()
        self.ax_rgb.clear()
        
        # Display single band
        band_data = self.img.read_band(self.current_band)
        self.ax_band.imshow(self.normalize_band(band_data), cmap='viridis')
        self.ax_band.set_title(f'Band {self.current_band + 1} - {self.img.bands.centers[self.current_band]:.2f}nm')
        
        # Display RGB composite
        rgb = self.create_rgb()
        self.ax_rgb.imshow(rgb)
        self.ax_rgb.set_title('RGB Composite\nR:650nm, G:550nm, B:450nm')
        
        # Remove axes for image displays
        self.ax_band.set_axis_off()
        self.ax_rgb.set_axis_off()
        
        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        if event.inaxes == self.ax_band and event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.img.shape[1] and 0 <= y < self.img.shape[0]:
                # Clear previous spectral plot
                self.ax_spec.clear()
                
                # Get spectral profile
                spectrum = self.img.read_pixel(y, x)
                wavelengths = self.img.bands.centers
                
                # Plot spectrum
                self.ax_spec.plot(wavelengths, spectrum)
                self.ax_spec.set_xlabel('Wavelength (nm)')
                self.ax_spec.set_ylabel('Radiance')
                self.ax_spec.set_title(f'Spectral Profile at ({x}, {y})')
                self.ax_spec.grid(True)
                
                self.fig.canvas.draw_idle()

def view_hyperspectral_data(data_file, header_file):
    viewer = HyperspectralViewer(data_file, header_file)
    viewer.fig.canvas.mpl_connect('button_press_event', viewer.on_click)
    plt.show()

# Example usage:

data_file = "afx102_1_2026_reflectance.dat"
header_file = "afx102_1_2026_reflectance.dat.hdr"
view_hyperspectral_data(data_file, header_file)
