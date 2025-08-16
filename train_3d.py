# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader

def main():

    args = cfg.parse_args()

    # Parse distributed GPUs
    if args.distributed and args.distributed.lower() != "none":
        gpu_ids = [int(gpu_id) for gpu_id in args.distributed.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        world_size = len(gpu_ids)
        print(f"Using GPUs: {gpu_ids}")
    else:
        gpu_ids = [0]  # Default to GPU 0
        world_size = 1
        print("Using single GPU mode (default GPU 0).")

    # Initialize model
    GPUdevice = torch.device(f"cuda:{gpu_ids[0]}")
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net.to(dtype=torch.bfloat16)

    if world_size > 1:
        # Wrap the model with DataParallel for multiple GPUs
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    if args.pretrain:
        print(f"Loading pretrained weights from {args.pretrain}")
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights, strict=False)
    sam_layers = (
                  []
                #   + list(net.image_encoder.parameters())
                #   + list(net.sam_prompt_encoder.parameters())
                  + list(net.sam_mask_decoder.parameters())
                  )
    mem_layers = (
                  []
                  + list(net.obj_ptr_proj.parameters())
                  + list(net.memory_encoder.parameters())
                  + list(net.memory_attention.parameters())
                  + list(net.mask_downsample.parameters())
                  )
    if len(sam_layers) == 0:
        optimizer1 = None
    else:
        optimizer1 = optim.Adam(sam_layers, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    if len(mem_layers) == 0:
        optimizer2 = None
    else:
        optimizer2 = optim.Adam(mem_layers, lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader, nice_test_loader, nice_val_loader = get_dataloader(args)
  
    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0
    
    for epoch in range(settings.EPOCH):

        # if epoch < 5:
        #     tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        #     logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        net.train()
        time_start = time.time()
        loss, prompt_loss, non_prompt_loss = function.train_sam(args, net, optimizer1, optimizer2, nice_train_loader, epoch)
        
        logger.info(f'Train loss: {loss}, {prompt_loss}, {non_prompt_loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            # Optional: Run additional validation on the validation set
            logger.info(f'Running validation set evaluation for epoch {epoch}')
            with torch.no_grad():
                for batch in nice_val_loader:
                    images = batch['image']
                    labels = batch['label']

                    if torch.cuda.is_available():
                        images = images.cuda()
                        # Iterate through each key-value pair in the nested dictionary
                        for k, v_dict in labels.items():
                                if isinstance(v_dict, dict):
                                    for m, v in v_dict.items():
                                        if isinstance(v, torch.Tensor):
                                            labels[k][m] = v.cuda()
                    outputs = net.propagate_in_video(images)
            torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))

    writer.close()


if __name__ == '__main__':
    main()