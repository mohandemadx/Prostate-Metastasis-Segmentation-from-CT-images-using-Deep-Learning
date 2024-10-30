"""
File: data_processing.py
Description: 

            
Author: Mohand Emad
Date: October 28, 2024
Version: 1.0
Contributions:
  - 
"""

# Main Logic 
import numpy as np
import pydicom
import os
import src

def load_dicom(dir):
    """
    Load DICOM files from the given directory and return them as a stacked matrix.
    Args:
        dir (str): Path to the directory containing DICOM files.
    Returns:
        np.array: Matrix of flattened DICOM images.
    """
    dicom_images = []
    for filename in os.listdir(dir):
        if filename.endswith('.dcm'):
            file_path = os.path.join(dir, filename)
            dicom_image = pydicom.dcmread(file_path)
            dicom_images.append(dicom_image.pixel_array)  # Access the pixel data

    # Flatten each image and stack into a matrix
    image_matrix = np.vstack([image.flatten() for image in dicom_images])
    
    return image_matrix, dicom_images


# Function to extract prostate contour information from RTSTRUCT FILE
def extract_prostate_contour(file, dicom_images):
    prostate_contours = []
    
    rtstruct = pydicom.dcmread(file)
    
    # Step 1: Find the ROI corresponding to the "Prostate"
    for roi in rtstruct.StructureSetROISequence:
        if roi.ROIName.lower() == "prostate":  # Adjust the name if needed
            prostate_roi_number = roi.ROINumber
            print(f"Found Prostate ROI with number: {prostate_roi_number}")
            break
    else:
        raise ValueError("Prostate ROI not found in the RTSTRUCT file.")

    # Step 2: Find the corresponding contours for the prostate ROI
    for roi_contour in rtstruct.ROIContourSequence:
        if roi_contour.ReferencedROINumber == prostate_roi_number:
            # Step 3: Extract ContourImageSequence for the prostate contours
            for contour in roi_contour.ContourSequence:
                # Each contour might be associated with a different image slice
                for image_sequence in contour.ContourImageSequence:
                    ref = image_sequence.ReferencedSOPInstanceUID
                    contour_points = contour.ContourData
                    number_of_points = contour.NumberOfContourPoints
                    
                    # # Ensure the DICOM image exists
                    # if ref not in dicom_images:
                    #     raise ValueError(f"DICOM image with SOPInstanceUID {ref} not found.")
                    
                    # Extract relevant DICOM image info
                    dicom_image = dicom_images[ref]
                    image_position = dicom_image.ImagePositionPatient  # (X0, Y0, Z0) position in patient coordinates
                    image_orientation = dicom_image.ImageOrientationPatient  # (Xr, Yr, Zr, Xc, Yc, Zc) row and column direction cosines
                    pixel_spacing = dicom_image.PixelSpacing  # [row_spacing, column_spacing]
                    index = dicom_image.InstanceNumber
                    
                    # Map the contour points to image coordinates
                    image_coordinates = map_contour_to_image(contour_points, image_position, image_orientation, pixel_spacing)
                    
                    prostate_contours.append({
                        "Ref": ref,
                        "ImageSequence": image_sequence,
                        "ContourData": contour_points,
                        "NumberOfPoints": number_of_points,
                        "ContourPixelData": image_coordinates
                    })
    
    return prostate_contours, index


def create_mask(contour, rows, columns):
    image_shape = (rows, columns)
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Separate the i and j coordinates from contour tuples
    i_coords, j_coords = zip(*contour_pixels)
    i_coords = np.array(i_coords)
    j_coords = np.array(j_coords)
    
    # Generate the polygon mask
    rr, cc = polygon(i_coords, j_coords, mask.shape)
    mask[rr, cc] = 1  # Set pixels inside contour to 1
    
    return mask

def create_mask_matrix():
    pass
    

    