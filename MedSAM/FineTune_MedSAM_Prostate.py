import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
import SimpleITK as sitk
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import measure
import sys

# Add MedSAM model path
sys.path.append("/home/jovyan/Data/CISC867/Code/MedSAM/MedSAM")
from segment_anything.build_sam import sam_model_registry
from monai.losses import DiceLoss  # MONAI's Dice Loss

# Paths and configurations
train_dir = "/home/jovyan/Data/CISC867/Data/Image/Or_patient_imagemask/STraining"
val_dir = "/home/jovyan/Data/CISC867/Data/Image/Or_patient_imagemask/TValidation"
test_dir = "/home/jovyan/Data/CISC867/Data/Image/Or_patient_imagemask/STesting"
output_dir = "/home/jovyan/Data/CISC867/Result/Medsam/Pretrain"
learning_rate = 1e-4
num_epochs = 100
batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
os.makedirs(output_dir, exist_ok=True)

# Utility: Generate bounding box from a mask with expansion
def get_bounding_box_from_mask(mask, expansion_mm=10, pixel_spacing=0.5):
    mask = mask.squeeze()
    contours = measure.find_contours(mask, 0.5)
    if contours:
        min_row, min_col = float("inf"), float("inf")
        max_row, max_col = -float("inf"), -float("inf")
        for contour in contours:
            row_min, col_min = contour.min(axis=0)
            row_max, col_max = contour.max(axis=0)
            min_row = min(min_row, row_min)
            min_col = min(min_col, col_min)
            max_row = max(max_row, row_max)
            max_col = max(max_col, col_max)
        expansion_px = int(expansion_mm / pixel_spacing)
        return [
            max(0, int(min_col) - expansion_px),
            max(0, int(min_row) - expansion_px),
            min(mask.shape[1], int(max_col) + expansion_px),
            min(mask.shape[0], int(max_row) + expansion_px),
        ]
    return None

# Dataset for prostate patient images and masks
class ProstatePatientDataset(Dataset):
    def __init__(self, patient_dir, target_size=(256, 256)):
        self.patient_dirs = glob.glob(os.path.join(patient_dir, "*"))
        self.target_size = target_size

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_path = self.patient_dirs[idx]
        image_files = sorted(glob.glob(os.path.join(patient_path, "*.nrrd")))
        mask_files = sorted(glob.glob(os.path.join(patient_path, "*.nii.gz")))

        images, masks = [], []
        for img_file, mask_file in zip(image_files, mask_files):
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))

            assert img.shape == mask.shape, f"Shape mismatch: {img.shape} vs {mask.shape}"

            for i in range(img.shape[0]):
                img_slice = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
                mask_slice = mask[i]

                img_resized = resize(img_slice, self.target_size, preserve_range=True, anti_aliasing=True)
                mask_resized = resize(mask_slice, self.target_size, preserve_range=True, anti_aliasing=False)

                images.append(torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0))
                masks.append(torch.tensor(mask_resized, dtype=torch.float32).unsqueeze(0))

        return torch.stack(images), torch.stack(masks)

# Data loaders
train_dataset = ProstatePatientDataset(train_dir)
val_dataset = ProstatePatientDataset(val_dir)
test_dataset = ProstatePatientDataset(test_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load MedSAM model
MedSAM_CKPT_PATH = "/home/jovyan/Data/CISC867/Code/MedSAM/medsam_vit_b.pth"
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, medsam_model.parameters()), lr=learning_rate)
dice_loss = DiceLoss(sigmoid=True)

# Training and validation loop
train_loss_curve, val_loss_curve = [], []

for epoch in range(num_epochs):
    medsam_model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        for img, mask in zip(images, masks):
            for slice_index in range(mask.shape[0]):
                mask_slice = mask[slice_index]

                if torch.sum(mask_slice) == 0:  # Skip empty slices
                    continue

                bbox = get_bounding_box_from_mask(mask_slice.cpu().numpy())
                if bbox is None:  # Skip invalid bounding boxes
                    continue

                input_data = {
                    "image": img[slice_index].unsqueeze(0).repeat(1, 3, 1, 1).to(device),
                    "original_size": img[slice_index].shape[-2:],
                    "boxes": torch.tensor([bbox], dtype=torch.float32).to(device),
                }

                output = medsam_model([input_data], multimask_output=False)
                preds = output[0]["masks"].squeeze(0)
                slice_loss = dice_loss(preds, mask_slice)

                slice_loss.backward()  # Backpropagation
                optimizer.step()
                optimizer.zero_grad()
                train_loss += slice_loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_loss_curve.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

    # Validation phase
    medsam_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            for img, mask in zip(images, masks):
                for slice_index in range(mask.shape[0]):
                    mask_slice = mask[slice_index]

                    if torch.sum(mask_slice) == 0:
                        continue

                    bbox = get_bounding_box_from_mask(mask_slice.cpu().numpy())
                    if bbox is None:
                        continue

                    input_data = {
                        "image": img[slice_index].unsqueeze(0).repeat(1, 3, 1, 1).to(device),
                        "original_size": img[slice_index].shape[-2:],
                        "boxes": torch.tensor([bbox], dtype=torch.float32).to(device),
                    }

                    output = medsam_model([input_data], multimask_output=False)
                    preds = output[0]["masks"].squeeze(0)
                    slice_loss = dice_loss(preds, mask_slice)
                    val_loss += slice_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_loss_curve.append(avg_val_loss)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

# Plot and save loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_loss_curve, label="Train Loss")
plt.plot(val_loss_curve, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
print(f"Saved loss curve to {output_dir}")

# Test the model
medsam_model.eval()
test_dsc = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        for img, mask in zip(images, masks):
            bbox = get_bounding_box_from_mask(mask.cpu().numpy())
            if bbox is None:
                continue

            input_data = {
                "image": img,
                "original_size": img.shape[-2:],
                "boxes": [torch.tensor(bbox).to(device)],
            }
            preds = medsam_model([input_data], multimask_output=False)["masks"][0]
            test_dsc += dice_loss(preds, mask).item()

avg_test_dsc = test_dsc / len(test_loader)
print(f"Final Test DSC = {avg_test_dsc:.4f}")
