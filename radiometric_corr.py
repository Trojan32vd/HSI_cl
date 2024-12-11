import numpy as np
import pandas as pd
from datetime import datetime
import os

def read_header(header_file):
    """
    Read ENVI header file and parse all metadata
    
    Parameters:
    -----------
    header_file : str
        Path to header file
        
    Returns:
    --------
    dict : Header information including wavelengths and metadata
    """
    header_info = {}
    current_array = None
    array_data = []
    
    with open(header_file, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle array continuation
            if current_array and '}' not in line:
                array_data.extend([x.strip() for x in line.split(',')])
                continue
            elif current_array and '}' in line:
                array_data.extend([x.strip() for x in line.split('}')[0].split(',')])
                # Convert array data to floats
                try:
                    header_info[current_array] = [float(x) for x in array_data if x]
                except ValueError:
                    header_info[current_array] = array_data
                current_array = None
                array_data = []
                continue
            
            if '=' in line:
                key, value = [x.strip() for x in line.split('=', 1)]
                key = key.lower()
                
                # Handle array start
                if '{' in value and '}' not in value:
                    current_array = key
                    array_data = [x.strip() for x in value.split('{')[1].split(',')]
                    continue
                # Handle single line arrays
                elif '{' in value and '}' in value:
                    value = value.split('{')[1].split('}')[0].strip()
                    if ',' in value:
                        try:
                            value = [float(x.strip()) for x in value.split(',') if x.strip()]
                        except ValueError:
                            value = [x.strip() for x in value.split(',') if x.strip()]
                
                # Convert numeric values
                if key in ['samples', 'lines', 'bands', 'data type']:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                
                header_info[key] = value
    
    return header_info

def apply_enhanced_radiometric_correction(reflectance_files, radiance_files, chunk_size=500):
    """
    Apply radiometric correction using both reflectance and original radiance data
    
    Parameters:
    -----------
    reflectance_files : dict
        Dictionary containing paths to reflectance input files
    radiance_files : dict
        Dictionary containing paths to original radiance files
    chunk_size : int
        Size of chunks for processing large datasets
    """
    print("Starting enhanced radiometric correction...")
    
    # Read both headers
    print("Reading header files...")
    refl_header = read_header(reflectance_files['header'])
    rad_header = read_header(radiance_files['header'])
    
    # Verify dimensions match
    if (refl_header['samples'] != rad_header['samples'] or 
        refl_header['lines'] != rad_header['lines'] or 
        refl_header['bands'] != rad_header['bands']):
        raise ValueError("Dimension mismatch between reflectance and radiance data")
    
    nbands = refl_header['bands']
    nlines = refl_header['lines']
    nsamples = refl_header['samples']
    
    print(f"Data dimensions: {nbands} bands, {nlines} lines, {nsamples} samples")
    
    # Extract wavelength information
    wavelengths = rad_header.get('wavelength', None)
    if wavelengths:
        print(f"Found {len(wavelengths)} wavelength values")
    
    # Create output file
    output_file = reflectance_files['data'].replace('reflectance', 'radcorr')
    print(f"Creating output file: {output_file}")
    
    # Create memory-mapped output file
    output_data = np.memmap(output_file, dtype=np.float32, mode='w+',
                           shape=(nbands, nlines, nsamples))
    
    # Get radiance scale factor from header
    radiance_scale = 1000.0  # Default scale factor
    if 'description' in rad_header:
        desc = rad_header['description']
        if isinstance(desc, str):
            desc = desc.lower()
            if 'mw' in desc and '*' in desc:
                try:
                    scale_str = desc.split('*')[-1].split(']')[0].strip()
                    radiance_scale = float(scale_str)
                    print(f"Found radiance scale factor: {radiance_scale}")
                except:
                    print("Warning: Could not parse radiance scale factor from header")
        elif isinstance(desc, list):
            # Handle case where description is a list
            desc_str = ' '.join(str(x) for x in desc).lower()
            if 'mw' in desc_str and '*' in desc_str:
                try:
                    scale_str = desc_str.split('*')[-1].split(']')[0].strip()
                    radiance_scale = float(scale_str)
                    print(f"Found radiance scale factor: {radiance_scale}")
                except:
                    print("Warning: Could not parse radiance scale factor from header")
    
    # Process data in chunks
    print("Processing data in chunks...")
    refl_data = np.memmap(reflectance_files['data'], dtype=np.float32, mode='r',
                         shape=(nbands, nlines, nsamples))
    
    for band in range(nbands):
        print(f"Processing band {band + 1}/{nbands}")
        
        # Calculate band-specific correction factor
        if wavelengths and band < len(wavelengths):
            # Wavelength-dependent correction factor
            wavelength = wavelengths[band]
            correction_factor = 1.0 + (wavelength - 500) / 1000.0
        else:
            correction_factor = 1.0
        
        for chunk_start in range(0, nlines, chunk_size):
            chunk_end = min(chunk_start + chunk_size, nlines)
            
            # Read chunk
            chunk = refl_data[band, chunk_start:chunk_end, :]
            
            # Apply radiometric correction
            corrected_chunk = chunk * correction_factor
            
            # Clip values to valid range [0,1]
            corrected_chunk = np.clip(corrected_chunk, 0, 1)
            
            # Save chunk
            output_data[band, chunk_start:chunk_end, :] = corrected_chunk
            
            # Flush changes
            output_data.flush()
    
    # Create header file for output
    output_header = output_file + '.hdr'
    print(f"Saving header file: {output_header}")
    
    with open(output_header, 'w') as f:
        f.write("ENVI\n")
        f.write("description = {Radiometrically corrected reflectance data}\n")
        f.write(f"samples = {nsamples}\n")
        f.write(f"lines = {nlines}\n")
        f.write(f"bands = {nbands}\n")
        f.write("header offset = 0\n")
        f.write("file type = ENVI Standard\n")
        f.write("data type = 4\n")
        f.write("interleave = bsq\n")
        f.write("byte order = 0\n")
        
        # Copy wavelength information
        if wavelengths:
            f.write("wavelength = {\n")
            f.write(',\n'.join(f" {w:0.6f}" for w in wavelengths))
            f.write("}\n")
        
        if 'wavelength units' in rad_header:
            f.write(f"wavelength units = {rad_header['wavelength units']}\n")
            
        # Copy additional metadata
        if 'acquisition date' in rad_header:
            f.write(f"acquisition date = {rad_header['acquisition date']}\n")
        if 'sensor type' in rad_header:
            f.write(f"sensor type = {rad_header['sensor type']}\n")
    
    print("Radiometric correction completed successfully!")
    return output_file

if __name__ == '__main__':
    # Define input files
    reflectance_files = {
        'data': 'afx102_1_2026_reflectance.dat',
        'header': 'afx102_1_2026_reflectance.dat.hdr'
    }
    
    radiance_files = {
        'data': 'afx102_1_2026_radiance.dat',
        'header': 'afx102_1_2026_radiance.hdr'
    }
    
    try:
        output_file = apply_enhanced_radiometric_correction(reflectance_files, radiance_files)
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")