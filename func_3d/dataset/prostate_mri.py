import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from PIL import Image
from func_3d.utils import random_click, generate_bbox


def resize_or_pad_mask(mask, target_shape):
    """
    Resize or pad a mask to match the target shape.

    Args:
        mask (np.ndarray): The input mask array with shape (depth, height, width).
        target_shape (tuple): The target shape (depth, height, width).

    Returns:
        np.ndarray: The resized or padded mask.
    """
    current_depth, current_height, current_width = mask.shape
    target_depth, target_height, target_width = target_shape

    # Handle depth adjustment
    if current_depth < target_depth:
        depth_padding = target_depth - current_depth
        mask = np.pad(mask, ((0, depth_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    elif current_depth > target_depth:
        mask = mask[:target_depth]

    resized_mask = []
    for slice_idx in range(mask.shape[0]):
        slice_img = Image.fromarray(mask[slice_idx])
        resized_slice = slice_img.resize((target_width, target_height), Image.NEAREST)
        resized_mask.append(np.array(resized_slice))

    return np.stack(resized_mask)


class ProstateMRI(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None, variation=0):
        self.name_list = os.listdir(os.path.join(data_path, mode, 'images'))
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        # Get the image and mask paths
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'images', name)
        mask_name = name.replace(".nrrd", ".nii.gz")  # Replace .nrrd with .nii.gz for the corresponding mask
        mask_path = os.path.join(self.data_path, self.mode, 'masks', mask_name)

        # Load image and mask
        img_sitk = sitk.ReadImage(img_path)
        mask_sitk = sitk.ReadImage(mask_path)
        img_np = sitk.GetArrayFromImage(img_sitk)
        mask_np = sitk.GetArrayFromImage(mask_sitk)
        print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}, min: {img_np.min()}, max: {img_np.max()}")
        print(f"Mask shape: {mask_np.shape}, dtype: {mask_np.dtype}, min: {mask_np.min()}, max: {mask_np.max()}")
        num_frames = img_np.shape[0]
        
        if self.video_length is None:
            video_length = int(num_frames / 4)
        else:
            video_length = self.video_length

        if num_frames > video_length and self.mode == 'Training':
            start_frame = np.random.randint(0, num_frames - video_length + 1)
        else:
            start_frame = 0

        # Initialize tensors and dictionaries
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        # Process each frame
        for i, frame_index in enumerate(range(start_frame, start_frame + video_length)):
            img_slice = img_np[frame_index]
            mask_slice = mask_np[frame_index]
            # Normalize and resize image
            img = Image.fromarray((img_slice * 255).astype(np.uint8)).convert('RGB')
            img = img.resize(newsize)
            
            img = torch.tensor(np.array(img)).float().permute(2, 0, 1)
            img_tensor[i, :, :, :] = img
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-7)

            # Resize and process mask
            obj_mask = Image.fromarray(mask_slice).resize(newsize, Image.NEAREST)
            obj_mask = torch.tensor(np.array(obj_mask)).float().unsqueeze(0)  # Add channel dimension [1, H, W]
            mask_dict[frame_index - start_frame] = {'mask': obj_mask}


            # Generate prompts (click or bbox)
            if self.prompt == 'click':
                point_label, pt = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=self.seed)
                pt_dict[frame_index - start_frame] = {'pt': pt}
                point_label_dict[frame_index - start_frame] = {'point_label': point_label}
            elif self.prompt == 'bbox':
                bbox = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
                if bbox is not None:
                    bbox_dict[frame_index - start_frame] = {'bbox': bbox}
        for frame_index, bbox_data in bbox_dict.items():
                    for obj_id, bbox in bbox_data.items():
                        if bbox.dtype != torch.float32:
                            bbox_dict[frame_index][obj_id] = torch.tensor(bbox, dtype=torch.float32)
        print(f"[DEBUG] label keys: {mask_dict.keys() if isinstance(mask_dict, dict) else label}")
        print(f"[DEBUG] bbox keys: {bbox_dict.keys() if isinstance(bbox_dict, dict) else bbox}")
        image_meta_dict = {'filename_or_obj': name}

        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
            }
