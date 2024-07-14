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
from torchvision.utils import save_image
import numpy as np
import cv2

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


INFERENCE_NO_SPN_FOLDER = "pt_inference_no_spn"
GT_DISP_FOLDER = prep_dataset_finetune.VALIDATION_DISPARITY_FOLDER

# TO-DO => 
# - interpolate vs upsample
# - fix torchvision
# - fix spn package

# def save_batch_images(output: torch.Tensor, stage: int, output_folder: str):s
def save_batch_images(outputs, stage: int, output_folder: str):
    
    utils_anynet.delete_folders([output_folder])
    utils_anynet.create_folders([output_folder])

    logging.warning(f"outputs.shape: {outputs.shape}")
    
    for i, output in enumerate(outputs):
        # Save each image as a PNG file
        output = output.squeeze(0)
        output = output * 255.0
        logging.info(f"output.shape: {output.shape}")
        mn = torch.min(output)
        mx = torch.max(output)
        # logging.info(f"mn: {mn} mx: {mx}")

        output_normalized = (output - mn) / (mx - mn)
        output_normalized = output_normalized * 255.0   
        # output_normalized = output_normalized.to(torch.uint8)

        output_np = output_normalized.cpu().detach().numpy()

        # Convert the data type to uint8
        output_np_uint8 = np.clip(output_np, 0, 255).astype('uint8')

        # Use OpenCV to write the image to a file
        cv2.imwrite(f"{output_folder}/image_{i}.png", output_np_uint8)
        # save_image(output_normalized, os.path.join(output_folder, f"image_{i}.png"))


def inference():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath,log, args.split_file)

    logging.warning(f"len(test_left_img): {len(test_left_img)}")

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    # return

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
    
    cudnn.benchmark = True
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
                logging.warning(f"[Stage {stage}] type(output): {type(output)} output.shape: {output.shape}")
            
            logging.info("[after squeezing]")
            for x in range(stages):
                # output = torch.squeeze(outputs[x], 1)
                # D1s[x].update(error_estimating(output, disp_L).item())
                # save_batch_images(output, stage=x, output_folder= INFERENCE_NO_SPN_FOLDER)
                # outputs_cpu = outputs[x].cpu()
                # save_batch_images(outputs_cpu, stage=x, output_folder= INFERENCE_NO_SPN_FOLDER)
                save_batch_images(outputs[x], stage=x, output_folder= INFERENCE_NO_SPN_FOLDER)
                logging.warning(f"[Stage {x}] output.dtype: {output.dtype} output.shape: {output.shape}")

        
        
        
        # info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
        # logging.info(f'Average test 3-Pixel Error = {info_str}')

        # info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])
        logging.error(f"=====================================================\n")
        
    #     log.info('[{}/{}] {}'.format(
    #         batch_idx, length_loader, info_str))

    # info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    # log.info('Average test 3-Pixel Error = ' + info_str)


    # args.start_epoch = 0
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         log.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         log.info("=> loaded checkpoint '{}' (epoch {})"
    #                  .format(args.resume, checkpoint['epoch']))
    #     else:
    #         log.info("=> no checkpoint found at '{}'".format(args.resume))
    #         log.info("=> Will start from scratch.")
    # else:
    #     log.info('Not Resume')
    # cudnn.benchmark = True
    # start_full_time = time.time()
    # if args.evaluate:
    #     test(TestImgLoader, model, log)
    #     return

    # for epoch in range(args.start_epoch, args.epochs):
    #     log.info('This is {}-th epoch'.format(epoch))
    #     adjust_learning_rate(optimizer, epoch)

    #     train(TrainImgLoader, model, optimizer, log, epoch)

    #     savefilename = args.save_path + '/checkpoint.tar'
    #     torch.save({
    #         'epoch': epoch,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }, savefilename)

    #     if epoch % 1 ==0:
    #         test(TestImgLoader, model, log)

    # test(TestImgLoader, model, log)
    # log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


# def main():
#     global args
#     log = logger.setup_logger(args.save_path + '/training.log')

#     train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
#         args.datapath,log, args.split_file)

#     TrainImgLoader = torch.utils.data.DataLoader(
#         DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
#         batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

#     TestImgLoader = torch.utils.data.DataLoader(
#         DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
#         batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

#     if not os.path.isdir(args.save_path):
#         os.makedirs(args.save_path)
#     for key, value in sorted(vars(args).items()):
#         log.info(str(key) + ': ' + str(value))

#     model = models.anynet.AnyNet(args)
#     model = nn.DataParallel(model).cuda()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
#     log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

#     if args.pretrained:
#         if os.path.isfile(args.pretrained):
#             checkpoint = torch.load(args.pretrained)
#             model.load_state_dict(checkpoint['state_dict'], strict=False)
#             log.info("=> loaded pretrained model'{}'"
#                      .format(args.pretrained))
#         else:
#             log.info("=> no pretrained model found at '{}'".format(args.pretrained))
#             log.info("=> Will start from scratch.")
#     args.start_epoch = 0
#     if args.resume:
#         if os.path.isfile(args.resume):
#             log.info("=> loading checkpoint '{}'".format(args.resume))
#             checkpoint = torch.load(args.resume)
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             log.info("=> loaded checkpoint '{}' (epoch {})"
#                      .format(args.resume, checkpoint['epoch']))
#         else:
#             log.info("=> no checkpoint found at '{}'".format(args.resume))
#             log.info("=> Will start from scratch.")
#     else:
#         log.info('Not Resume')
#     cudnn.benchmark = True
#     start_full_time = time.time()
#     if args.evaluate:
#         test(TestImgLoader, model, log)
#         return

#     for epoch in range(args.start_epoch, args.epochs):
#         log.info('This is {}-th epoch'.format(epoch))
#         adjust_learning_rate(optimizer, epoch)

#         train(TrainImgLoader, model, optimizer, log, epoch)

#         savefilename = args.save_path + '/checkpoint.tar'
#         torch.save({
#             'epoch': epoch,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#         }, savefilename)

#         if epoch % 1 ==0:
#             test(TestImgLoader, model, log)

#     test(TestImgLoader, model, log)
#     log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        if batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)



def inference_test(dataloader, model, log):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())

        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)


def test(dataloader, model, log):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())

        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)


def error_estimating(disp, ground_truth, maxdisp=192):
    # logging.info("[error_estimating] -> entering")
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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