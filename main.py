import src.data_processing as dp

contour_path = 'Prostate Dataset/Prostate-AEC-001/11-17-1992-NA-RX SIMULATION-82988/0.000000-Contouring-60430/1-1.dcm'
image_dir = 'Prostate Dataset/Prostate-AEC-001/11-17-1992-NA-RX SIMULATION-82988/2.000000-Pelvis-13578'


# Load the DICOM images and find the index of the reference image
image_data = dp.load_dicom(image_dir)


