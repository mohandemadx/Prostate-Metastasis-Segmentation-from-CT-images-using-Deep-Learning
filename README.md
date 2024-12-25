# Prostate Metastasis Segmentation from CT Images using Deep Learning

This repository implements a deep learning pipeline for segmenting prostate metastasis from CT images. The process leverages Python, MONAI, Torch, and CUDA to build an efficient and accurate segmentation system.

## Tools and Libraries
- **Python**: Primary programming language.
- **MONAI**: Medical Open Network for AI library for deep learning in medical imaging.
- **Torch**: PyTorch deep learning framework.
- **CUDA**: GPU acceleration for faster processing.

---

## Steps

### 1. DATA Preparation
Prepare the dataset by organizing CT images and corresponding segmentation masks. Ensure the following:
- Data is stored in a structured format (e.g., NIFTI, DICOM).
- Dataset is split into training, validation, and testing subsets.

### 2. Preprocessing
Perform preprocessing steps to normalize and augment the data:
- Resample images to a consistent voxel spacing.
- Normalize intensity values.
- Apply data augmentation techniques like flipping, rotation, and cropping to increase dataset diversity.

### 3. Training Pipeline
Train the deep learning model using MONAI and Torch:
- Define the segmentation model architecture.
- Set hyperparameters (learning rate, batch size, number of epochs).
- Utilize GPU acceleration with CUDA for training.
- Monitor training metrics (e.g., loss, Dice coefficient).

### 4. Evaluation using Dice Coefficient
Evaluate the model performance:
- Use the Dice Similarity Coefficient to assess segmentation accuracy.
- Compare predictions against ground truth masks.
- Adjust the model as needed based on evaluation results.

### 5. Saving Segmentation to RT-STRUCT using Postprocessing Pipeline
Generate RT-STRUCT files from the segmentation results:
- Post-process the segmented images (e.g., smoothing, contour extraction).
- Convert segmentation masks to RT-STRUCT format for clinical use.
- Save the output in a standardized directory structure.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/Prostate-Metastasis-Segmentation.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Follow the steps outlined above to prepare data, train the model, and evaluate results.

---

## Acknowledgments
This project leverages MONAI and PyTorch, with significant contributions from the medical imaging and deep learning communities.

