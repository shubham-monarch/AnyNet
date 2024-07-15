import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn

import models.anynet
# from torchvision.utils import save_image
from torchvision.utils import save_image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# custom imports
import logging, coloredlogs
import prep_dataset_finetune
import utils_anynet

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')


args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls

GT_DISP_FOLDER = prep_dataset_finetune.VALIDATION_DISPARITY_FOLDER
PT_INFERENCE_NO_SPN_DIR = "pt_inference_no_spn"
HISTOGRAMS_NO_SPN_DIR = f"{PT_INFERENCE_NO_SPN_DIR}/histograms" 

# TO-DO => 
# - interpolate vs upsample [undo this / research this]
# - fix torchvision
# - fix spn package
# - negative disparity
# - go through git issues
# - go through the paper once
# - compare s1 vs s2 vs s3
    # - accuracy not improving across stage
    # - check if pre-trained model is loaded
    # - check if there is a pytorch version issue while loading the pre-trained model 


def tensorf32_to_histogram(output: torch.Tensor, output_dir: str, stage: int = None):
    utils_anynet.delete_folders([output_dir])
    utils_anynet.create_folders([output_dir])
    
    logging.info(f"[tensorf32_to_histogram] stage: {stage}")
    output_np = output.cpu().numpy()
    logging.info(f"[tensorf32_to_histogram] output_np.shape: {output_np.shape} output.shape: {output.shape}")
    for i in range(output.shape[0]):
        plt.figure()
        # Flatten the image to get the distribution of all pixel values
        img = output_np[i]
        logging.info(f"[tensorf32_to_histogram] img.shape: {img.shape}")
        img_flat = img.flatten()
        logging.info(f"mx: {np.max(img_flat)} mn: {np.min(img_flat)}")
        plt.hist(img_flat, bins=50, color='blue', alpha=0.7)
        plt.title(f"Histogram for Image {i}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        
        # Save the histogram
        plt.savefig(f"{output_dir}/histogram_{i}_{stage}.png")
        plt.close()
    

def tensorf32_to_png(output, stage: int, output_folder: str):
    
    utils_anynet.delete_folders([output_folder])
    utils_anynet.create_folders([output_folder])
    return

def inference():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath,log, args.split_file)

    logging.warning(f"len(test_left_img): {len(test_left_img)}")

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    # for key, value in sorted(vars(args).items()):
    #     log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        logging.error(f"args.pretrained: {args.pretrained}")
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model'{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    
    # cudnn.benchmark = True
    model.eval()

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    # length_loader = len(dataloader)

    # model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        
        logging.error(f"=====================================================")
        if batch_idx > 0: 
            logging.warning(f"Breaking after 3 batches")
            break
        logging.error(f"batch_idx: {batch_idx}")

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        # logging.info(f"disp_L.shape: {disp_L.shape} disp_L.dtype: {disp_L.dtype}")
        
        with torch.no_grad():
            outputs = model(imgL, imgR)
            logging.info(f"type(outputs): {type(outputs)} len(outputs): {len(outputs)}")
            logging.info(f"[before squeezing]")
            
            for stage, output in enumerate(outputs):
                logging.warning(f"[Stage {stage}] output.dtype: {output.dtype} output.shape: {output.shape}")
            
            # logging.info("[after squeezing]")
            for x in range(stages):
                # output = torch.squeeze(outpsuts[x], 1)
                # D1s[x].update(error_estimating(output, disp_L).item())
                # save_batch_images(output, stage=x, output_folder= INFERENCE_NO_SPN_FOLDER)
                # outputs_cpu = outputs[x].cpu()
                # save_batch_images(outputs_cpu, stage=x, output_folder= INFERENCE_NO_SPN_FOLDER)
                logging.warning(f"[Stage {x}] output.dtype: {output.dtype} output.shape: {output.shape}")
                tensorf32_to_histogram(output, output_dir=HISTOGRAMS_NO_SPN_DIR, stage=x)
        
        
        
        # info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
        # logging.info(f'Average test 3-Pixel Error = {info_str}')

        # info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])
        logging.error(f"=====================================================\n")
        
    #     log.info('[{}/{}] {}'.format(
    #         batch_idx, length_loader, info_str))

    # info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    # log.info('Average test 3-Pixel Error = ' + info_str)
   


def error_estimating(disp, ground_truth, maxdisp=192):
    # logging.info("[error_estimating] -> entering")
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    # main()
    coloredlogs.install(level="INFO", force=True)  # install a handler on the root logger
    inference()