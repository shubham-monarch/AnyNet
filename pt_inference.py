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
from PIL import Image

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

# gt disparity folder
GT_DISP_FOLDER = prep_dataset_finetune.VALIDATION_DISPARITY_FOLDER

# pt inference folder(s)
PT_INFERENCE_NO_SPN_DIR = "pt_inference_no_spn"
INPUT_IMAGES_NO_SPN_DIR = f"{PT_INFERENCE_NO_SPN_DIR}/input"
DISPARITY_IMAGES_NO_SPN_DIR = f"{PT_INFERENCE_NO_SPN_DIR}/disp"

# TO-DO => 
# [x]- interpolate vs upsample [undo this / research this] ==> UPSAMPING IS WRONG
# - fix torchvision
# - fix spn package
# - negative disparity
# - go through git issues
# - go through the paper once
# - compare s1 vs s2 vs s3
    # - accuracy not improving across stage
    # - check if pre-trained model is loaded
    # - check if there is a pytorch version issue while loading the pre-trained model 
    # - fix the s1, s2, s3 disparity maps
    # - is testloader loading images sequentially
    # - reduce 3 pixel error

        
def inference():

    FOLDERS_TO_CREATE = [PT_INFERENCE_NO_SPN_DIR, INPUT_IMAGES_NO_SPN_DIR, DISPARITY_IMAGES_NO_SPN_DIR]
    utils_anynet.delete_folders(FOLDERS_TO_CREATE)
    utils_anynet.create_folders(FOLDERS_TO_CREATE)

    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath,log, args.split_file)

    # logging.warning(f"len(test_left_img): {len(test_left_img)}")
    # logging.warning(f"type(test_left_img): {type(test_left_img)} type(test_left_img[0]): {type(test_left_img[0])}")

    
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

    # utils_anynet.delete_folders([HISTOGRAMS_NO_SPN_DIR])
    # utils_anynet.create_folders([HISTOGRAMS_NO_SPN_DIR])
    
    # cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)

    for batch_idx, (path_l , path_r, path_disp, imgL, imgR, disp_L) in enumerate(TestImgLoader):

        batch_cutoff = 0        
        if batch_idx > batch_cutoff:
            break
        logging.error(f"=====================[BATCH {batch_idx} STARTED]================================")
        # logging.warning(f"[{batch_idx}]")
        # logging.info(f"path_l: {path_l}")
        # logging.info(f"path_r: {path_r}")
        # logging.info(f"path_disp: {path_disp}\n")

        logging.info(f"type(imgL): {type(imgL)} imgL.shape: {imgL.shape} imgL.dtype: {imgL.dtype}")
        logging.info(f"type(imgL[0]): {type(imgL[0])} imgL[0].shape: {imgL[0].shape} imgL[0].dtype: {imgL[0].dtype}")
        # logging.info(f"type(imgR): {type(imgR)} type(imgR[0]): {type(imgR[0])} imgR[0].shape: {imgR[0].shape} imgR[0].dtype: {imgR[0].dtype}")
        logging.info(f"type(disp_L): {type(disp_L)}  disp_L.shape: {disp_L.shape} disp_L.dtype: {disp_L.dtype}")
        logging.info(f"type(disp_L[0]): {type(disp_L[0])} disp_L[0].shape: {disp_L[0].shape} disp_L[0].dtype: {disp_L[0].dtype}")
        # logging.info(f"type(disp_L): {type(disp_L)} type(disp_L[0]): {type(disp_L[0])} disp_L[0].shape: {disp_L[0].shape} disp_L[0].dtype: {disp_L[0].dtype}")
        
        # logging.info(f"type(imgR[0]): {type(imgR[0])} imgR[0].shape: {imgR[0].shape} imgR[0].dtype: {imgR[0].dtype}")
        # logging.info(f"type(disp_L[0]): {type(disp_L[0])} disp_L[0].shape: {disp_L[0].shape} disp_L[0].dtype: {disp_L[0].dtype}")
        # Convert to numpy and adjust dimensions
        np_images = imgL.cpu().numpy()
        np_images = np.transpose(np_images, (0, 2, 3, 1))  # Change from [N, C, H, W] to [N, H, W, C]

        # Scale to 0-255 and convert to uint8
        np_images = (np_images * 255).clip(0, 255).astype(np.uint8)

        # Save each image
        for i, img_array in enumerate(np_images):
            img = Image.fromarray(img_array)
            path_img = os.path.basename(path_l[i])
            img.save(f"{INPUT_IMAGES_NO_SPN_DIR}/{path_img}")

        # continue
        # hconcat_LR = cv2.hconcat([imgL[0].numpy(), imgR[0].numpy()])
        filename = os.path.basename(path_l[0])
        for idx, path in enumerate(path_l):
            filename = os.path.basename(path)
            # cv2.imwrite(f"{INPUT_IMAGES_NO_SPN_DIR}/{filename}", imgL[0].numpy())
        
        # continue
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
            
            logging.info("[after squeezing]")
            for x in range(stages):
                if x < 2:
                    continue
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())
                # save_batch_images(output, stage=x, output_folder= INFERENCE_NO_SPN_FOLDER)
                # outputs_cpu = outputs[x].cpu()
                # save_batch_images(outputs_cpu, stage=x, output_folder= INFERENCE_NO_SPN_FOLDER)
                logging.warning(f"[STAGE {x}]")
                logging.info(f"output.dtype: {output.dtype} output.shape: {output.shape}")
                # tensorf32_to_hisstogram(output, output_dir=HISTOGRAMS_NO_SPN_DIR, stage=x)
                # tensorf32_to_png(output, output_dir=HISTOGRAMS_NO_SPN_DIR, stage=x)
                
                # [8, W, H]
                output_cpu = output.cpu().numpy()
                for image_idx, output_cpu_ in enumerate(output_cpu):
                    # logging.info(f"[{image_idx}] --> output_np_.min(): {output_np_.min()} output_np_.max(): {output_np_.max()}")
                    
                    # [W,H]
                    # img_cpu = np.asarray(output.cpu())
                    # img_save = np.clip(output_cpu_, 0, 2**16)
                    # img_save = (img_save * 256.0).astype(np.uint16)
                    # img_name = f"{HISTOGRAMS_NO_SPN_DIR}/histogram_{batch_idx}_{image_idx}_{x}.png"
                    # cv2.imwrite(img_name, img_save)
                    
                    output_int8 = utils_anynet.uint8_normalization(output_cpu_)
                    path_disp = os.path.basename(path_l[image_idx])
                    cv2.imwrite(f"{DISPARITY_IMAGES_NO_SPN_DIR}/{path_disp}", output_int8)
                    # # pass

                    # logging.info(f"file_")

        info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
        logging.info(f'Average test 3-Pixel Error = {info_str}')

        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])
        
        logging.error(f"=====================[BATCH {batch_idx} FINISHED]================================\n")
        
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