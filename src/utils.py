import numpy as np

def map_contour_to_image(contour_data, image_position, image_orientation, pixel_spacing):
    """
    Maps 3D contour points (X, Y, Z) to 2D image coordinates (i, j).
    
    :param contour_data: List of (X, Y, Z) contour points
    :param image_position: [X0, Y0, Z0] position of the image top-left corner in patient space
    :param image_orientation: [Xr, Yr, Zr, Xc, Yc, Zc] row and column direction cosines
    :param pixel_spacing: [row_spacing, column_spacing] spacing between pixels in mm
    
    :return: List of (i, j) coordinates in the image
    """
    # Split image orientation into row and column direction cosines
    row_direction = np.array(image_orientation[:3])  # [Xr, Yr, Zr]
    col_direction = np.array(image_orientation[3:])  # [Xc, Yc, Zc]
    
    # Image position (top-left corner of the slice in patient coordinates)
    image_position = np.array(image_position)  # [X0, Y0, Z0]
    
    # Pixel spacing (mm per pixel)
    row_spacing, col_spacing = pixel_spacing
    
    # To store 2D image coordinates (i, j)
    image_coordinates = []
    
    # Loop through each contour point (X, Y, Z)
    for i in range(0, len(contour_data), 3):
        contour_point = np.array(contour_data[i:i+3])  # Extract X, Y, Z
        
        # Vector from image origin to contour point
        vector_to_point = contour_point - image_position
        
        # Project onto row and column directions
        i_coord = np.dot(vector_to_point, row_direction) / row_spacing
        j_coord = np.dot(vector_to_point, col_direction) / col_spacing
        
        # Append the (i, j) image coordinates
        image_coordinates.append((i_coord, j_coord))
    
    return image_coordinates