import numpy as np
import matplotlib.pyplot as plt
from hyppy.format.envi import Envi
from hyppy.plot import spectraplot, RGBplot
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class HyperspectralViewer:
    def __init__(self, root, data_file, header_file):
        self.root = root
        self.root.title("Hyperspectral Data Viewer")
        
        # Load the data
        self.envi_data = Envi(data_file, header_file)
        self.data = self.envi_data.load()
        self.wavelengths = self.envi_data.waves
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure
        self.fig = plt.Figure(figsize=(10, 8))
        
        # Add canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create control panel
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add view selection
        ttk.Label(self.control_frame, text="View:").pack(side=tk.LEFT)
        self.view_var = tk.StringVar(value="RGB")
        self.view_combo = ttk.Combobox(
            self.control_frame, 
            textvariable=self.view_var,
            values=["RGB", "Single Band", "Spectral Profile"],
            state="readonly"
        )
        self.view_combo.pack(side=tk.LEFT, padx=5)
        self.view_combo.bind('<<ComboboxSelected>>', self.update_view)
        
        # Add band selection
        ttk.Label(self.control_frame, text="Band:").pack(side=tk.LEFT)
        self.band_var = tk.StringVar(value="1")
        self.band_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.band_var,
            values=[str(i+1) for i in range(self.data.shape[2])],
            state="readonly"
        )
        self.band_combo.pack(side=tk.LEFT, padx=5)
        self.band_combo.bind('<<ComboboxSelected>>', self.update_view)
        
        # Initialize with RGB view
        self.update_view()
        
    def show_rgb(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Select bands for RGB (approximate visible wavelengths)
        red_idx = np.argmin(np.abs(self.wavelengths - 650))
        green_idx = np.argmin(np.abs(self.wavelengths - 550))
        blue_idx = np.argmin(np.abs(self.wavelengths - 450))
        
        rgb = np.dstack([
            self.normalize_band(self.data[:,:,red_idx]),
            self.normalize_band(self.data[:,:,green_idx]),
            self.normalize_band(self.data[:,:,blue_idx])
        ])
        
        ax.imshow(rgb)
        ax.set_title("RGB Composite")
        ax.axis('off')
        self.canvas.draw()
        
    def show_single_band(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        band_idx = int(self.band_var.get()) - 1
        band_data = self.normalize_band(self.data[:,:,band_idx])
        
        im = ax.imshow(band_data, cmap='viridis')
        ax.set_title(f"Band {band_idx + 1} ({self.wavelengths[band_idx]:.1f} nm)")
        self.fig.colorbar(im)
        ax.axis('off')
        self.canvas.draw()
        
    def show_spectral_profile(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Get center pixel
        center_y, center_x = self.data.shape[0]//2, self.data.shape[1]//2
        spectrum = self.data[center_y, center_x, :]
        
        ax.plot(self.wavelengths, spectrum)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Spectral Profile (Center Pixel: {center_y}, {center_x})")
        ax.grid(True)
        self.canvas.draw()
        
    def normalize_band(self, band):
        """Normalize band data to 0-1 range"""
        min_val = np.percentile(band, 2)
        max_val = np.percentile(band, 98)
        if max_val == min_val:
            return np.zeros_like(band)
        return np.clip((band - min_val) / (max_val - min_val), 0, 1)
    
    def update_view(self, event=None):
        view = self.view_var.get()
        if view == "RGB":
            self.show_rgb()
        elif view == "Single Band":
            self.show_single_band()
        elif view == "Spectral Profile":
            self.show_spectral_profile()

def main():
    root = tk.Tk()
    root.geometry("800x600")
    
    # Replace these with your actual file paths
    data_file = "afx102_1_2026_radcorr.dat"
    header_file = "afx102_1_2026_radcorr.dat.hdr"
    
    app = HyperspectralViewer(root, data_file, header_file)
    root.mainloop()

if __name__ == "__main__":
    main()