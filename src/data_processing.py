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
import torch
from utils import map_contour_to_image

def load_dicom(dir):
    # Get a list of DICOM file paths in the directory
    dicom_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.dcm')]
    
    if not dicom_files:
        raise ValueError("No DICOM files found in the directory")

    # Load the first image to determine (H, W)
    first_image = pydicom.dcmread(dicom_files[0])
    H, W = first_image.pixel_array.shape
    D = len(dicom_files)
    C = 1  # Grayscale

    # Initialize the array for DICOM images
    dicom_images = np.empty((D, C, H, W), dtype=first_image.pixel_array.dtype)
    image_data = []

    # Load each DICOM image into the preallocated array
    for i, file_path in enumerate(dicom_files):
        dicom_image = pydicom.dcmread(file_path)
        uid = dicom_image.SOPInstanceUID
        image_position = dicom_image.ImagePositionPatient  # (X0, Y0, Z0) position in patient coordinates
        image_orientation = dicom_image.ImageOrientationPatient  # (Xr, Yr, Zr, Xc, Yc, Zc) row and column direction cosines
        pixel_spacing = dicom_image.PixelSpacing 
        image_data.append({
                        "SliceIndex": i,
                        "UID": uid,
                        "ImagePosition": image_position,
                        "ImageOrientation": image_orientation,
                        "PixelSpacing": pixel_spacing
                    })  
        
        # Add the image data to the array, with a channel dimension
        dicom_images[i, 0, :, :] = dicom_image.pixel_array

    # Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.from_numpy(dicom_images)

    return image_tensor, image_data


# Function to extract prostate contour information
def extract_prostate_contour(rtstruct_path, image_data):
    contours_data = []
    contours = []
    rtstruct = pydicom.dcmread(rtstruct_path)
    
    # Step 1: Find the ROI corresponding to the "Prostate"
    for roi in rtstruct.StructureSetROISequence:
        if roi.ROIName.lower() == "prostate":  # Adjust the name if needed
            prostate_roi_number = roi.ROINumber
            print(f"Found Prostate ROI with number: {prostate_roi_number}")
            break
    else:
        raise ValueError("Prostate ROI not found in the RTSTRUCT file.")

    for roi_contour in rtstruct.ROIContourSequence:
        if roi_contour.ReferencedROINumber == prostate_roi_number:
            for contour in roi_contour.ContourSequence:
                # Directly access the first entry in ContourImageSequence
                image_sequence = contour.ContourImageSequence[0]
                uid = image_sequence.ReferencedSOPInstanceUID  # Directly get the first UID
                contour_points = contour.ContourData
                number_of_points = contour.NumberOfContourPoints
                
                index = next((i for i, item in enumerate(image_data) if item["UID"] == uid), None)
                
                if index is None:
                    raise ValueError(f"DICOM image with SOPInstanceUID {uid} not found.")
                
                # Extract relevant DICOM image info
                data = image_data[index]
                image_position = data['ImagePosition'] # (X0, Y0, Z0) position in patient coordinates
                image_orientation = data['ImageOrientation']  # (Xr, Yr, Zr, Xc, Yc, Zc) row and column direction cosines
                pixel_spacing = data['PixelSpacing']  # [row_spacing, column_spacing]

                # Map the contour points to image coordinates
                image_coordinates = map_contour_to_image(contour_points, image_position, image_orientation, pixel_spacing)
                
                # Store or process the contour data as needed
                contours_data.append({
                    "UID": uid,
                    "index": index,
                    "ContourData": contour_points,
                    "NumberOfPoints": number_of_points,
                    "ContourPixelData": image_coordinates
                })
                contours.insert(index, image_coordinates)
    
    return contours, contours_data


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
    

    