import os
import glob
import csv
import numpy as np
import SimpleITK as sitk

# Helper functions
def dice_coefficient(mask1, mask2):
    """
    Calculate the Dice Coefficient between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    return (2.0 * intersection) / union if union != 0 else 1.0

def jaccard_index(mask1, mask2):
    """
    Calculate the Jaccard Index (IoU) between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    return intersection / union if union != 0 else 1.0

# Directory paths
mask_dir = "/home/jovyan/Data/CISC867/Code/MedSAM/MedSAM_output_5mm_1times"
ground_truth_mask_dir = "/home/jovyan/Data/CISC867/Data/Image/T2_prostate_mask"
output_dir = "/home/jovyan/Data/CISC867/Result/Medsam/Pretrain"
output_csv_path = os.path.join(output_dir, "mask_comparison_results_onlymask.csv")

# Initialize results list
results = []

# Iterate through all mask files
for mask in glob.glob(os.path.join(mask_dir, "*.nii.gz")):
    mask = mask.replace('\\', '/')  # Ensure consistent path format
    print(f"Processing file: {mask}")
    target_file = os.path.basename(mask)
    
    # Locate the corresponding ground truth file
    ground_truth = os.path.join(ground_truth_mask_dir, target_file.replace('_medsam_final.nii.gz', '.nii.gz'))
    
    if not os.path.exists(ground_truth):
        print(f"Ground truth file not found: {ground_truth}")
        continue
    
    # Read the mask and ground truth images
    mask_image = sitk.ReadImage(mask)
    ground_truth_image = sitk.ReadImage(ground_truth)
    mask_array = sitk.GetArrayFromImage(mask_image)
    ground_truth_array = sitk.GetArrayFromImage(ground_truth_image)
    
    # Ensure binary masks
    mask_array = (mask_array > 0).astype(np.int32)
    ground_truth_array = (ground_truth_array > 0).astype(np.int32)
    
    # Skip if both masks are empty
    if np.all(mask_array == 0) and np.all(ground_truth_array == 0):
        print(f"Skipping {mask}: Both mask and ground truth are empty.")
        continue
    
    # Calculate metrics
    dice = dice_coefficient(mask_array, ground_truth_array)
    jaccard = jaccard_index(mask_array, ground_truth_array)
    
    # Store results
    results.append([os.path.basename(mask), os.path.basename(ground_truth), dice, jaccard])

# Save the results to a CSV file
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Mask', 'Ground truth', 'Dice Coefficient', 'IoU'])  # Write header
    writer.writerows(results)  # Write data rows

print(f"Results saved to {output_csv_path}")
