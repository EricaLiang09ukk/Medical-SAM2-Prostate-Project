from .btcv import BTCV
from .amos import AMOS
from .prostate_mri import ProstateMRI
from .prostate_mri_new import ProstateMRIDataset, ProstateMRIAligned
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torch
import numpy as np

def custom_collate_fn(batch):
    filtered_batch = []

    for item in batch:
        if item is None:
            continue

        # Check for empty dictionaries in masks or other fields
        valid = True
        for label in item['label']:
            if not item['label'][label]:
                print("LABEL", label)
                item['label'][label][1] = torch.zeros(1, 1024, 1024)
                
        for bbox in item['bbox']:
            if not item['bbox'][bbox]:
                item['bbox'][bbox]["bbox"] = np.array(torch.tensor(np.array([0,0,0,0])))

        if valid:
            filtered_batch.append(item)
        else:
            print(f"[WARNING] Skipping sample with empty masks or bbox: {item}")

    if not filtered_batch:
        print("[WARNING] Skipping empty batch.")
        return None

    # Use the default_collate for valid items
    return torch.utils.data.default_collate(filtered_batch)

def get_dataloader(args):
    transform_train = transforms.Compose([
         transforms.Resize((args.image_size,args.image_size)),
         transforms.ToTensor(),
     ])

    transform_train_seg = transforms.Compose([
         transforms.Resize((args.out_size,args.out_size)),
         transforms.ToTensor(),
     ])

    transform_test = transforms.Compose([
         transforms.Resize((args.image_size, args.image_size)),
         transforms.ToTensor(),
     ])

    transform_test_seg = transforms.Compose([
         transforms.Resize((args.out_size,args.out_size)),
         transforms.ToTensor(),
     ])
    
    if args.dataset == 'btcv':
        # BTCV dataset
        train_dataset = BTCV(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Training', prompt=args.prompt)
        test_dataset = BTCV(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Test', prompt=args.prompt)
        val_dataset = BTCV(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Validation', prompt=args.prompt)

    elif args.dataset == 'prostate_mri':
        # Prostate MRI dataset
        train_dataset = ProstateMRI(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Training', prompt=args.prompt)
        test_dataset = ProstateMRI(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Test', prompt=args.prompt)
        val_dataset = ProstateMRI(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Validation', prompt=args.prompt)
    
    elif args.dataset == 'prostate_mri_new':
        # Prostate MRI dataset
        train_dataset = ProstateMRIDataset(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Training', prompt=args.prompt)
        test_dataset = ProstateMRIDataset(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Test', prompt=args.prompt)
        val_dataset = ProstateMRIDataset(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Validation', prompt=args.prompt)
    else:
        raise ValueError("The dataset is not supported now!")

    # Use a subset for debugging if needed
    if args.debug:
        subset_size = args.subset_size if hasattr(args, 'subset_size') else 10
        train_dataset = Subset(train_dataset, list(range(min(len(train_dataset), subset_size))))
        val_dataset = Subset(val_dataset, list(range(min(len(val_dataset), subset_size))))
        test_dataset = Subset(test_dataset, list(range(min(len(test_dataset), subset_size))))
    
    weights = []
    for item in train_dataset:
        # Extract mask_dict from the dataset item
        mask_dict = item['label']
        has_object = any(mask['mask'].sum() > 0 for mask in mask_dict.values())  # Check if any slice has objects
        weights.append(1.0 if not has_object else 10.0)  # Adjust weight as needed

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, pin_memory=False, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)  # Use a smaller batch size for testing/validation

    return train_loader, test_loader, val_loader