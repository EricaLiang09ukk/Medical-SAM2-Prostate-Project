import numpy as np
import os
import torch
from skimage import transform, measure
import SimpleITK as sitk
import glob
import torch.nn.functional as F
import sys

# Add MedSAM model path to system
sys.path.append("D:/CISC867/MedSAM/MedSAM")
from segment_anything.build_sam import sam_model_registry

# Environment configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Perform MedSAM inference on a given image
@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device).unsqueeze(0).unsqueeze(0)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    return (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

# Generate bounding box from mask with expansion
def get_bounding_box_from_mask(mask, expansion_mm=10, pixel_spacing=0.5):
    contours = measure.find_contours(mask, 0.5)
    if contours:
        min_row, min_col = np.min(contours[0], axis=0)
        max_row, max_col = np.max(contours[0], axis=0)
        expansion_px = int(expansion_mm / pixel_spacing)
        min_row = max(0, int(min_row) - expansion_px)
        min_col = max(0, int(min_col) - expansion_px)
        max_row = min(mask.shape[0], int(max_row) + expansion_px)
        max_col = min(mask.shape[1], int(max_col) + expansion_px)
        return [min_col, min_row, max_col, max_row]
    return None

# Save a volume as a NIfTI file with reference metadata
def save_nifti(mask_volume, reference_image_path, output_path):
    reference_image = sitk.ReadImage(reference_image_path)
    mask_image = sitk.GetImageFromArray(mask_volume)
    mask_image.SetSpacing(reference_image.GetSpacing())
    mask_image.SetOrigin(reference_image.GetOrigin())
    mask_image.SetDirection(reference_image.GetDirection())
    sitk.WriteImage(mask_image, output_path)

# Directory paths for input and output
nrrd_dir = "D:/CISC867/PKG - Prostate-MRI-US-Biopsy-STL/T2MR_nrrd"
mask_dir = "D:/CISC867/PKG - Prostate-MRI-US-Biopsy-STL/T2_prostate_mask"
output_dir = "D:/CISC867/MedSAM/MedSAM_output_5mm_1times"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
if device == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Allocated:", torch.cuda.memory_allocated(0))
    print("Memory Cached:", torch.cuda.memory_reserved(0))

# Load MedSAM model
MedSAM_CKPT_PATH = "/home/jovyan/Data/CISC867/Code/MedSAM/medsam_vit_b.pth"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

# Process NRRD files
for nrrd_file in glob.glob(os.path.join(nrrd_dir, "*.nrrd")):
    nrrd_file = nrrd_file.replace('\\', '/')
    print(f"Processing NRRD file: {nrrd_file}")
    
    # Load mask file corresponding to the NRRD file
    mask_file = os.path.join(mask_dir, os.path.splitext(os.path.basename(nrrd_file))[0] + ".nii.gz")
    if not os.path.exists(mask_file):
        print(f"Mask file not found for {nrrd_file}")
        continue
    
    # Read mask and input image
    mask_sitk = sitk.ReadImage(mask_file)
    mask_np = sitk.GetArrayFromImage(mask_sitk)
    pixel_spacing = mask_sitk.GetSpacing()[0]
    
    img_sitk = sitk.ReadImage(nrrd_file)
    img_np = sitk.GetArrayFromImage(img_sitk)
    mask_volume = np.zeros_like(img_np, dtype=np.uint8)

    # Iterate through each slice of the 3D volume
    for slice_idx in range(img_np.shape[0]):
        img_slice = img_np[slice_idx]
        reference_mask_slice = mask_np[slice_idx]

        if np.max(reference_mask_slice) == 1:  # Check if the slice has mask values
            bbox = get_bounding_box_from_mask(reference_mask_slice, expansion_mm=5, pixel_spacing=pixel_spacing)
            if bbox is None:
                continue

            # Normalize and prepare slice for MedSAM input
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            img_slice = (img_slice * 255).astype(np.uint8)
            img_3c = np.repeat(img_slice[:, :, None], 3, axis=-1)

            # Resize image and bounding box
            img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            scaled_bbox = [int(b * 1024 / img_slice.shape[1]) for b in bbox]
            
            # Perform MedSAM inference
            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_1024_tensor)
            initial_seg = medsam_inference(medsam_model, image_embedding, scaled_bbox, 1024, 1024)
            
            # Resize segmentation result back to original dimensions
            final_seg_resized = transform.resize(initial_seg, img_slice.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            mask_volume[slice_idx] = final_seg_resized

    # Save final segmentation as NIfTI
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(nrrd_file))[0]}_medsam_final.nii.gz")
    output_path = output_path.replace('\\', '/')
    save_nifti(mask_volume, nrrd_file, output_path)

    print(f"Saved segmentation to {output_path}")
