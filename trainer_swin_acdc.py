import logging
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage.interpolation import zoom
import numpy as np

from utils import DiceLoss, test_single_volume


def trainer_acdc(args, model, snapshot_path):
    """Train Swin-Unet model on ACDC dataset"""
    from TransUnet_acdc_supplymentary.dataset_acdc import BaseDataSets, RandomGenerator
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # Load ACDC dataset
    db_train = BaseDataSets(base_dir=args.root_path, split='train', list_dir=None,
                            transform=RandomGenerator(output_size=[args.img_size, args.img_size]))
    db_val = BaseDataSets(base_dir=args.root_path, split='val', list_dir=None,
                          transform=RandomGenerator(output_size=[args.img_size, args.img_size]))
    
    logging.info("The length of train set is: {}".format(len(db_train)))
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, 
                            num_workers=0, pin_memory=False)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    print("{} iterations per epoch. {} max iterations".format(len(train_loader), max_iterations))
    
    best_loss = 10e10
    best_dice = 0.0
    # Validation mỗi N epoch (tùy chỉ) mỗi N epoch thì sẽ validate thêm Dice + HD95
    eval_interval = 10  # Validate every N epochs
    
    for epoch_num in range(max_epoch):
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            
            # Ensure correct image size (224, 224)
            if image_batch.shape[-2:] != (args.img_size, args.img_size):
                # Resize to (224, 224)
                B, C, H, W = image_batch.shape
                image_batch_np = image_batch.cpu().numpy()
                label_batch_np = label_batch.cpu().numpy()
                
                image_resized = []
                label_resized = []
                for b in range(B):
                    img_b = zoom(image_batch_np[b], (C, args.img_size / H, args.img_size / W), order=3)
                    lbl_b = zoom(label_batch_np[b], (args.img_size / H, args.img_size / W), order=0)
                    image_resized.append(img_b)
                    label_resized.append(lbl_b)
                
                image_batch = torch.from_numpy(np.array(image_resized)).float()
                label_batch = torch.from_numpy(np.array(label_resized)).long()
            
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Learning rate scheduling
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            
            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        
        # Calculate epoch average loss
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num + 1, batch_loss, batch_ce_loss, batch_dice_loss))
        print(f'Train epoch: {epoch_num + 1}/{max_epoch} : loss : {batch_loss:.6f}, loss_ce: {batch_ce_loss:.6f}, loss_dice: {batch_dice_loss:.6f}')
        
        # Validation every N epochs (with Dice/HD95 metrics)
        if (epoch_num + 1) % eval_interval == 0:
            model.eval()
            
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), total=len(val_loader),
                                                   leave=False, desc="Validation"):
                    image_batch = sampled_batch['image'].squeeze(0).cpu().detach().numpy()
                    label_batch = sampled_batch['label'].squeeze(0).cpu().detach().numpy()
                    
                    # Resize to model input size (224, 224)
                    img_h, img_w = image_batch.shape[-2:]
                    if img_h != args.img_size or img_w != args.img_size:
                        image_batch = zoom(image_batch, (args.img_size / img_h, args.img_size / img_w), order=3)
                        label_batch = zoom(label_batch, (args.img_size / img_h, args.img_size / img_w), order=0)
                    
                    metric_i = test_single_volume(torch.from_numpy(image_batch).unsqueeze(0).unsqueeze(0), 
                                                  torch.from_numpy(label_batch).unsqueeze(0),
                                                  model, classes=num_classes,
                                                  patch_size=[args.img_size, args.img_size])
                    if i_batch == 0:
                        metric_list = np.array(metric_i)
                    else:
                        metric_list = np.vstack([metric_list, np.array(metric_i)])
            
            metric_list = np.mean(metric_list, axis=0)
            
            # Log metrics for each class
            for class_i in range(num_classes - 1):
                writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                  metric_list[class_i, 0], epoch_num + 1)
                writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                  metric_list[class_i, 1], epoch_num + 1)
            
            mean_dice = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar('info/val_mean_dice', mean_dice, epoch_num + 1)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, epoch_num + 1)
            
            logging.info('Val epoch: %d : mean_dice : %f, mean_hd95: %f' % (
                epoch_num + 1, mean_dice, mean_hd95))
            print(f'Val epoch: {epoch_num + 1}/{max_epoch} : mean_dice : {mean_dice:.6f}, mean_hd95: {mean_hd95:.6f}')
            
            # Save best model based on Dice
            if mean_dice > best_dice:
                save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                best_dice = mean_dice
                logging.info("save best model to {}".format(save_mode_path))
        
        # Validation every epoch (loss-based, for quick feedback)
        else:
            model.eval()
            val_batch_dice_loss = 0
            val_batch_ce_loss = 0
            
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), total=len(val_loader),
                                                   leave=False, desc="Val loss"):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    
                    val_batch_dice_loss += loss_dice.item()
                    val_batch_ce_loss += loss_ce.item()

                val_batch_ce_loss /= len(val_loader)
                val_batch_dice_loss /= len(val_loader)
                val_batch_loss = 0.4 * val_batch_ce_loss + 0.6 * val_batch_dice_loss
                
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                    epoch_num + 1, val_batch_loss, val_batch_ce_loss, val_batch_dice_loss))
                print(f'Val epoch: {epoch_num + 1}/{max_epoch} : loss : {val_batch_loss:.6f}, loss_ce: {val_batch_ce_loss:.6f}, loss_dice: {val_batch_dice_loss:.6f}')
                
                # Save best model based on loss
                if val_batch_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = val_batch_loss
                    logging.info("save best model to {}".format(save_mode_path))
        
        # Save checkpoint every epoch
        save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num + 1}.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save checkpoint to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"
