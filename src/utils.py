import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt

# Function to convert a list of 

# Function to map contour points to image coordinates and create a binary mask
def map_contour_to_image_and_create_mask(contour_data, image_position, image_orientation, pixel_spacing, rows=1024, columns=1024):
    # Split image orientation into row and column direction cosines
    row_direction = np.array(image_orientation[:3])  # [Xr, Yr, Zr]
    col_direction = np.array(image_orientation[3:])  # [Xc, Yc, Zc]
    
    # Image position (top-left corner of the slice in patient coordinates)
    image_position = np.array(image_position)  # [X0, Y0, Z0]
    
    # Pixel spacing (mm per pixel)
    row_spacing, col_spacing = pixel_spacing
    
    # Initialize a blank image and a binary mask of size 1024x1024 with zeros
    image_array = np.zeros((rows, columns), dtype=np.float32)
    mask = np.zeros((rows, columns), dtype=np.float32)
    
    # Lists to store contour coordinates
    i_coords = []
    j_coords = []
    
    # Loop through each contour point (X, Y, Z)
    for i in range(0, len(contour_data), 3):
        contour_point = np.array(contour_data[i:i+3])  # Extract X, Y, Z
        
        # Vector from image origin to contour point
        vector_to_point = contour_point - image_position
        
        # Project onto row and column directions
        i_coord = int(np.dot(vector_to_point, row_direction) / row_spacing)
        j_coord = int(np.dot(vector_to_point, col_direction) / col_spacing)
        
        # Check if the coordinates are within the bounds of the image
        if 0 <= i_coord < rows and 0 <= j_coord < columns:
            image_array[i_coord, j_coord] = 1  # Mark the contour point in the image array
            i_coords.append(i_coord)
            j_coords.append(j_coord)

    # Generate the polygon mask using the collected contour coordinates
    if i_coords and j_coords:  # Only create mask if there are valid coordinates
        rr, cc = polygon(i_coords, j_coords, mask.shape)
        mask[rr, cc] = 1  # Set pixels inside contour to 1
        # print(type(mask), mask.shape, mask.dtype)
    
    return image_array, mask