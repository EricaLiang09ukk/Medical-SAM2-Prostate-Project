import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from PIL import Image

from func_3d.utils import random_click, generate_bbox
import numpy as np
from PIL import Image

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
        # Pad depth
        depth_padding = target_depth - current_depth
        mask = np.pad(mask, ((0, depth_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    elif current_depth > target_depth:
        # Crop depth
        mask = mask[:target_depth]

    # Resize each slice to match target height and width
    resized_mask = []
    for slice_idx in range(mask.shape[0]):
        slice_img = Image.fromarray(mask[slice_idx])
        resized_slice = slice_img.resize((target_width, target_height), Image.NEAREST)  # Use NEAREST for masks
        resized_mask.append(np.array(resized_slice))

    return np.stack(resized_mask)

class ProstateMRIDataset(Dataset):
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
        newsize = (self.img_size, self.img_size)  # Ensure this matches the model's expected input size

        # Get the images and masks
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'images', name)
        mask_path = os.path.join(self.data_path, self.mode, 'masks', name)

        # Get all .dcm files in the image path
        img_files = sorted([f for f in os.listdir(img_path) if f.endswith('.dcm')])
        mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith('.dcm')])

        # Check if there is at least one valid mask
        for mask_file in mask_files:
            mask_dcm = pydicom.dcmread(os.path.join(mask_path, mask_file))
            if 'prostate' in str(mask_dcm.SeriesDescription):
                valid_masks = [mask_dcm.pixel_array.astype(np.uint8)]

        # Make sure we have the correct number of frames (or handle fewer frames)
        num_frames = valid_masks[0].shape[0]
        # video_length = self.video_length if self.video_length else num_frames
        if self.video_length is None:
            video_length = int(num_frames / 4)
        else:
            video_length = self.video_length
            
        if num_frames > video_length and self.mode == 'Training':
            start_frame = np.random.randint(0, num_frames - self.video_length + 1)
        else:
            start_frame = 0
        print(f"Processing patient {name} with {len(valid_masks)} valid masks.")
                
        # Initialize tensors and dictionaries
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        # Load all frames for the patient
        for i, frame_index in enumerate(range(start_frame, start_frame + video_length)):
            # Load the DICOM image
            img_dcm = pydicom.dcmread(os.path.join(img_path, img_files[frame_index]))
            img = img_dcm.pixel_array.astype(np.uint8)

            # Convert to PIL image and resize
            img = Image.fromarray(img).convert('RGB')
            img = img.resize(newsize)

            # Convert to tensor and adjust dimensions
            img = torch.tensor(np.array(img)).float().permute(2, 0, 1)

            img_tensor[i, :, :, :] = img

        # Load and process valid masks
        for mask in valid_masks:
            # Load the DICOM mask
            #mask_dcm = pydicom.dcmread(os.path.join(mask_path, mask_file))
            #mask = mask_dcm.pixel_array.astype(np.uint8)

            # Resize each slice to match [H*16, W*16]
            for slice_idx in range(start_frame, start_frame + video_length):
                mask_slice = mask[slice_idx, :, :]
                obj_mask = Image.fromarray(mask_slice).resize((self.img_size, self.img_size))
                obj_mask = torch.tensor(np.array(obj_mask)).float().unsqueeze(0)  # Add channel dimension [1, H*16, W*16]
                mask_dict[slice_idx - start_frame] = {'mask': obj_mask}

                # Generate prompts (click or bbox)
                if self.prompt == 'click':
                    point_label, pt = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=self.seed)
                    pt_dict[slice_idx - start_frame] = {'pt': pt}
                    point_label_dict[slice_idx - start_frame] = {'point_label': point_label}
                elif self.prompt == 'bbox':
                    bbox = generate_bbox(np.array(mask_slice).squeeze(), variation=self.variation, seed=self.seed)
                    if bbox is not None:
                        bbox_dict[slice_idx - start_frame] = {'bbox': bbox}
                else:
                    raise ValueError('Prompt not recognized')
        for frame_index, bbox_data in bbox_dict.items():
            for obj_id, bbox in bbox_data.items():
                if bbox.dtype != torch.float32:
                    bbox_dict[frame_index][obj_id] = torch.tensor(bbox, dtype=torch.float32)
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

        
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox


class ProstateMRIAligned(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None, variation=0):
        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'images'))
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        self.video_length = args.video_length if mode == 'Training' else None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)
        expected_depth = 60  # Expected consistent depth

        # Get the patient name and paths
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'images', name)
        mask_path = os.path.join(self.data_path, self.mode, 'masks', name)

        # Load image files
        img_files = sorted([f for f in os.listdir(img_path) if f.endswith('.dcm')])
        mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith('.dcm')])

        # Filter out masks that do not match the expected depth
#         valid_masks = []
        for mask_file in mask_files:
            mask_dcm = pydicom.dcmread(os.path.join(mask_path, mask_file))
            if 'prostate' in str(mask_dcm.SeriesDescription):
                masks = mask_dcm.pixel_array.astype(np.uint8)
#             if mask.shape[0] == expected_depth:
#                 valid_masks.append(mask)
#             else:
#                 print(f"Excluding mask {mask_file} with shape {mask.shape}")

        # If no valid masks remain, skip this patient
#         if not valid_masks:
#             print(f"No valid masks found for patient {name}. Skipping...")
#             return None
        
        # Stack the valid masks along the first axis
#         masks = np.stack(valid_masks, axis=0)  # Shape: (num_masks, frames, h, w)
        print("Masks SHape:", masks.shape)
        num_frames = masks.shape[0]
#         video_length = self.video_length if self.video_length else num_frames
        if self.video_length is None:
            video_length = int(num_frames / 4)
        else:
            video_length = self.video_length
            
        if num_frames > video_length and self.mode == 'Training':
            start_frame = np.random.randint(0, num_frames - 4)
        else:
            start_frame = 0

        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(start_frame, start_frame + video_length):
            if frame_index >= len(img_files):
                continue  # Skip if image is missing
            print("FRAME INDEX:", frame_index, "START_FRAME:", start_frame)
            
            # Load and process 2D image
            img_dcm = pydicom.dcmread(os.path.join(img_path, img_files[frame_index]))
            img = img_dcm.pixel_array.astype(np.uint8)
            img = Image.fromarray(img).convert('RGB').resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float()

            # Load and process corresponding 3D mask slice
            mask_slice = masks[frame_index - start_frame, :, :]
            print("Mask slice", masks.shape)# Extract 2D slice for all masks
            obj_list = np.unique(mask_slice[mask_slice > 0])
            print("Mask Slice:", np.unique(mask_slice))
            diff_obj_mask_dict = {}
            diff_obj_bbox_dict = {} if self.prompt == 'bbox' else None
            diff_obj_pt_dict = {} if self.prompt == 'click' else None
            diff_obj_point_label_dict = {} if self.prompt == 'click' else None
            bbox_list = []
            for obj in obj_list:
                obj_mask = (mask_slice == obj).astype("uint8")
                print("Object Mask", obj_mask.shape)
                obj_mask = Image.fromarray(obj_mask).resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).float()

                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    pt, label = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=self.seed)
                    diff_obj_pt_dict[obj] = pt
                    diff_obj_point_label_dict[obj] = label

                elif self.prompt == 'bbox':
                    bbox = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
                    print("BBOX", bbox)
                    if bbox is not None:
                        bbox_list.append(bbox)
#                         diff_obj_bbox_dict[obj] = torch.tensor(bbox).float()
                            
            diff_obj_bbox_dict['bbox'] = np.array(bbox_list)
            img_tensor[int(frame_index - start_frame)] = img
            mask_dict[int(frame_index - start_frame)] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[int(frame_index - start_frame)] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[int(frame_index - start_frame)] = diff_obj_pt_dict
                point_label_dict[int(frame_index - start_frame)] = diff_obj_point_label_dict
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

