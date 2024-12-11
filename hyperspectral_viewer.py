import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

class HyperspectralViewer:
    def __init__(self, data_file, header_file):
        """Initialize the hyperspectral data viewer"""
        self.load_header(header_file)
        self.load_data(data_file)
        self.create_gui()
        
    def load_header(self, header_file):
        """Load and parse the ENVI header file"""
        self.header = {}
        with open(header_file, 'r') as f:
            lines = f.readlines()
            
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if '=' in line:
                if current_key:
                    self.header[current_key] = self.process_value(current_value)
                    current_value = []
                
                key, value = [x.strip() for x in line.split('=', 1)]
                current_key = key.lower()
                
                if '{' in value and '}' in value:
                    value = value.split('{')[1].split('}')[0].strip()
                    self.header[current_key] = self.process_value([value])
                    current_key = None
                elif '{' in value:
                    current_value.append(value.split('{')[1])
                else:
                    self.header[current_key] = self.process_value([value])
                    current_key = None
            elif current_key:
                if '}' in line:
                    current_value.append(line.split('}')[0])
                    self.header[current_key] = self.process_value(current_value)
                    current_key = None
                else:
                    current_value.append(line)
                    
    def process_value(self, value_list):
        """Process header values, converting to appropriate types"""
        value = ', '.join(value_list).strip()
        
        try:
            if ',' in value:
                return [float(x.strip()) for x in value.split(',') if x.strip()]
            else:
                return int(value)
        except ValueError:
            return value
            
    def load_data(self, data_file):
        """Load hyperspectral data using memory mapping"""
        self.samples = self.header['samples']
        self.lines = self.header['lines']
        self.bands = self.header['bands']
        
        self.data = np.memmap(data_file, dtype=np.float32, mode='r',
                            shape=(self.bands, self.lines, self.samples))
        
        self.wavelengths = self.header.get('wavelength', 
                                         np.arange(self.bands))
                                         
    def enhance_image(self, img):
        """Enhance image contrast using percentile normalization"""
        p2, p98 = np.percentile(img, (2, 98))
        img_enhanced = np.clip((img - p2) / (p98 - p2), 0, 1)
        return img_enhanced
        
    def create_gui(self):
        """Create the main GUI window with visualization panels"""
        self.root = tk.Tk()
        self.root.title("Hyperspectral Data Viewer")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create figure for visualization
        self.fig = Figure(figsize=(12, 8))
        
        # Create canvas
        canvas = tkagg.FigureCanvasTkAgg(self.fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = tkagg.NavigationToolbar2Tk(canvas, main_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.ax_rgb = self.fig.add_subplot(221)  # RGB composite
        self.ax_band = self.fig.add_subplot(222)  # Single band
        self.ax_spectrum = self.fig.add_subplot(212)  # Spectrum plot
        
        # Add slider for band selection
        self.slider_frame = ttk.Frame(main_frame)
        self.slider_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.slider_frame, text="Band:").pack(side=tk.LEFT)
        self.band_slider = ttk.Scale(self.slider_frame, from_=0, 
                                   to=self.bands-1, orient=tk.HORIZONTAL,
                                   command=self.update_band_display)
        self.band_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Create RGB band selection
        rgb_frame = ttk.Frame(main_frame)
        rgb_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(rgb_frame, text="R:").pack(side=tk.LEFT)
        self.r_slider = ttk.Scale(rgb_frame, from_=0, to=self.bands-1, 
                                orient=tk.HORIZONTAL,
                                command=lambda x: self.update_rgb_display())
        self.r_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(rgb_frame, text="G:").pack(side=tk.LEFT)
        self.g_slider = ttk.Scale(rgb_frame, from_=0, to=self.bands-1, 
                                orient=tk.HORIZONTAL,
                                command=lambda x: self.update_rgb_display())
        self.g_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(rgb_frame, text="B:").pack(side=tk.LEFT)
        self.b_slider = ttk.Scale(rgb_frame, from_=0, to=self.bands-1, 
                                orient=tk.HORIZONTAL,
                                command=lambda x: self.update_rgb_display())
        self.b_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add enhancement controls
        enhance_frame = ttk.Frame(main_frame)
        enhance_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.enhance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(enhance_frame, text="Enhance Contrast", 
                       variable=self.enhance_var,
                       command=self.update_rgb_display).pack(side=tk.LEFT)
        
        # Set default RGB bands for natural-looking visualization
        default_bands = {
            'r': min(self.bands - 1, 80),  # Around 680nm
            'g': min(self.bands - 1, 50),  # Around 550nm
            'b': min(self.bands - 1, 20)   # Around 450nm
        }
        
        self.r_slider.set(default_bands['r'])
        self.g_slider.set(default_bands['g'])
        self.b_slider.set(default_bands['b'])
        
        # Initial display
        self.update_band_display(0)
        self.update_rgb_display()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.fig.tight_layout()
        
    def update_band_display(self, band_index):
        """Update the single band display"""
        band_index = int(band_index)
        band_data = self.data[band_index].copy()
        
        if self.enhance_var.get():
            band_data = self.enhance_image(band_data)
        
        self.ax_band.clear()
        im = self.ax_band.imshow(band_data, cmap='gray')
        self.ax_band.set_title(f'Band {band_index} '
                              f'({self.wavelengths[band_index]:.2f} nm)')
        
        # Add colorbar
        divider = make_axes_locatable(self.ax_band)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.fig.colorbar(im, cax=cax)
        
        self.fig.canvas.draw_idle()
        
    def update_rgb_display(self):
        """Update the RGB composite display"""
        try:
            r_idx = int(self.r_slider.get())
            g_idx = int(self.g_slider.get())
            b_idx = int(self.b_slider.get())
            
            # Create RGB composite
            rgb = np.dstack((
                self.data[r_idx].copy(),
                self.data[g_idx].copy(),
                self.data[b_idx].copy()
            ))
            
            # Apply enhancement if enabled
            if self.enhance_var.get():
                rgb = np.dstack([self.enhance_image(rgb[:,:,i]) for i in range(3)])
            
            self.ax_rgb.clear()
            self.ax_rgb.imshow(rgb)
            self.ax_rgb.set_title(f'RGB Composite (R:{r_idx}, G:{g_idx}, B:{b_idx})')
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error updating RGB display: {str(e)}")
                    
    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes in [self.ax_rgb, self.ax_band]:
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                if 0 <= x < self.samples and 0 <= y < self.lines:
                    self.plot_spectrum(x, y)
                    
    def plot_spectrum(self, x, y):
        """Plot spectrum for selected pixel"""
        spectrum = self.data[:, y, x]
        
        self.ax_spectrum.clear()
        self.ax_spectrum.plot(self.wavelengths, spectrum)
        self.ax_spectrum.set_xlabel('Wavelength (nm)')
        self.ax_spectrum.set_ylabel('Reflectance')
        self.ax_spectrum.set_title(f'Spectrum at pixel ({x}, {y})')
        self.ax_spectrum.grid(True)
        
        self.fig.canvas.draw_idle()
        
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == '__main__':
    # File paths
    data_file = 'afx102_1_2026_radcorr.dat'
    header_file = 'afx102_1_2026_radcorr.dat.hdr'
    
    try:
        viewer = HyperspectralViewer(data_file, header_file)
        viewer.run()
    except Exception as e:
        print(f"Error: {str(e)}")