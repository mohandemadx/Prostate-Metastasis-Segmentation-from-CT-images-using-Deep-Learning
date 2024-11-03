"""
File: data_processing.py
Description: 

            
Author: Mohand Emad
Date: November 3, 2024
Version: 2.0
Contributions:
  - load_dicom() ---> meta_data and pixel_array
  - extract_prostate_contour() ---> extract meta_data and mask
"""

# Main Logic 
import numpy as np
import pydicom
import os
from utils import map_contour_to_image_and_create_mask


def load_dicom(dir):
    '''
        "SliceIndex": i,
        "UID": uid,
        "ImagePosition": image_position,
        "ImageOrientation": image_orientation,
        "PixelSpacing": pixel_spacing,
        "PixelArray": normalized_pixel_array
    '''
    # Get a list of DICOM file paths in the directory
    dicom_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.dcm')]
    
    if not dicom_files:
        raise ValueError("No DICOM files found in the directory")

    image_data = []

    # Load each DICOM image into the preallocated array
    for i, file_path in enumerate(dicom_files):
        dicom_image = pydicom.dcmread(file_path)
        uid = dicom_image.SOPInstanceUID
        image_position = dicom_image.ImagePositionPatient  # (X0, Y0, Z0) position in patient coordinates
        image_orientation = dicom_image.ImageOrientationPatient  # (Xr, Yr, Zr, Xc, Yc, Zc) row and column direction cosines
        pixel_spacing = dicom_image.PixelSpacing 
        
        
        # Normalize the pixel array to the range [0, 1]
        pixel_array = dicom_image.pixel_array.astype(np.float32)  # Convert to float32 for normalization
        normalized_pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array) + 1e-6)  # Avoid division by zero
        # Add the image data to the array, with a channel dimension
        image_data.append({
                        "SliceIndex": i,
                        "UID": uid,
                        "ImagePosition": image_position,
                        "ImageOrientation": image_orientation,
                        "PixelSpacing": pixel_spacing,
                        "PixelArray": normalized_pixel_array
                    })  

    return image_data

# Function to extract prostate contour information
def extract_prostate_contour(rtstruct_path, image_data):
    '''
        "UID": uid,
        "Index": index,
        "ContourData": contour_points,
        "NumberOfPoints": number_of_points,
        "ContourPixelData": image_array,
        "Mask": mask
    '''
    contours_data = []
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
                
                index = None  # Initialize index to None
                for i, item in enumerate(image_data):

                    # Check if the UID in the item matches the specified uid
                    if item.get("UID") == uid:
                        index = i  # Set index to the current index
                        print(f"Match found at index {index} with UID: {item['UID']}")
                        break  # Stop searching after finding the first match

                # If no match is found, index remains None
                if index is None:
                    print("No matching UID found.")
                else:
                    print(f"UID found at index {index}")
                
                # Extract relevant DICOM image info
                data = image_data[index]
                image_position = data['ImagePosition'] # (X0, Y0, Z0) position in patient coordinates
                image_orientation = data['ImageOrientation']  # (Xr, Yr, Zr, Xc, Yc, Zc) row and column direction cosines
                pixel_spacing = data['PixelSpacing']  # [row_spacing, column_spacing]

                # Map the contour points to image coordinates
                image_array, mask = map_contour_to_image_and_create_mask(contour_points, image_position, image_orientation, pixel_spacing)
                
                # Store or process the contour data as needed
                contours_data.append({
                    "UID": uid,
                    "Index": index,
                    "ContourData": contour_points,
                    "NumberOfPoints": number_of_points,
                    "ContourPixelData": image_array,
                    "Mask": mask
                })
    
    return contours_data

    
