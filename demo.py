import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image, ImageOps
import torch.backends.cudnn as cudnn
import flow_viz
# from utils.utils import InputPadder
import torchvision.transforms as T
import models
from multiscaleloss import multiscaleEPE, realEPE




DEVICE = 'cuda'

def load_image(imfile, transform):
    image = Image.open(imfile)
    image = transform(image)
    img = np.array(image).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# class_boundary = list(np.arange(0, 16, 2))
# class_boundary.append(400)
# class_boundary = list(np.arange(0, 400, 400//10))
# class_boundary.append(400)

def viz(args, img1, flo, gt_flo, _id):
    # img = img1[0].permute(1,2,0).cpu().numpy()
    # img2 = img2[0].permute(1,2,0).cpu().numpy()
    gt_flo = gt_flo[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    gt_flo = flow_viz.flow_to_image(gt_flo)
    flo = flow_viz.flow_to_image(flo)
    try:
        os.mkdir(args.output_path)
    except:
        pass
    
    # mag = np.sqrt(np.sum(flo**2, axis=2)) 
    # _class = np.zeros(mag.shape)
    # for i in range(len(class_boundary) - 1):
    #     _class += np.where((class_boundary[i] < mag) & (mag < class_boundary[i + 1]), len(class_boundary) - i, 0)

    # _class = _class / (len(class_boundary) - 1) * 255

    # flox_rgb = Image.fromarray(_class.astype('uint8'), 'RGB')
    # flox_gray = ImageOps.grayscale(flox_rgb)    
    # flox_gray = Image.fromarray(_class.astype('uint8'), 'L')    

    flox_rgb = Image.fromarray(gt_flo.astype('uint8'), 'RGB')
    flox_rgb.save(args.output_path + '/attacked_flow_' + _id + '.png')
    flox_rgb = Image.fromarray(flo.astype('uint8'), 'RGB')
    flox_rgb.save(args.output_path + '/predicted_flow_' + _id + '.png')

    # flox_rgb = Image.fromarray(img.astype('uint8'), 'RGB')
    # flox_rgb.save(args.output_path + '/' + 'img1.png')
    # flox_rgb = Image.fromarray(img2.astype('uint8'), 'RGB')
    # flox_rgb.save(args.output_path + '/' + 'img2.png')

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 255)
    return perturbed_image

def demo(args):
    transform = T.Resize((320, 448))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    network_data = torch.load(args.pretrained)
    args.arch = network_data["arch"]
    model = models.__dict__[args.arch](network_data).to(device)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    model = model.module
    model.to(DEVICE)
    model.eval()

    # with torch.no_grad():
    _id = 0
    images = glob.glob(os.path.join(args.path, '*.png')) + \
                glob.glob(os.path.join(args.path, '*.jpg'))
    
    images = sorted(images)
    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        image1 = load_image(imfile1, transform)
        image2 = load_image(imfile2, transform)
        print(torch.max(image1), torch.min(image1))
        print(image1.shape)

        inp = [img1, img2]
        inp = torch.cat(inp, 1).to(device)


        output = model(inp)
        
        # start attack
        if args.attack_type != 'None':
            inp.requires_grad = True # for attack
        
        torch.cuda.empty_cache()

        ori = inp.data.clone().detach()
        flow_gt = output.clone().detach()

        if args.attack_type != 'None':
            if args.attack_type == 'FGSM':
                epsilon = args.epsilon
                pgd_iters = 1
            else:
                epsilon = 2.5 * args.epsilon / args.iters
                pgd_iters = args.iters
            
            shape = inp.shape
            delta = (np.random.rand(np.product(shape)).reshape(shape) - 0.5) * 2 
            inp.data = ori + torch.from_numpy(delta).type(torch.float).cuda()
            inp.data = torch.clamp(inp.data, 0.0, 255.0)
            output = model(inp)
        
            for iter in range(pgd_iters):
                flow = output
                flow2_EPE = args.div_flow * realEPE(output, target, sparse=args.sparse)
                model.zero_grad()
                inp.requires_grad = True
                flow2_EPE.backward()
                data_grad = inp.grad.data
                args.channel = int(args.channel)
                if args.channel == -1:
                    inp.data = fgsm_attack(inp, epsilon, data_grad)
                else:
                    inp.data[:, args.channel, :, :] = fgsm_attack(inp, epsilon, data_grad)[:, args.channel, :, :]
                if args.attack_type == 'PGD':
                    offset = inp.data - ori
                    inp.data = ori + torch.clamp(offset, -args.epsilon, args.epsilon)
            output = model(inp)
        viz(args, inp, output.detach(), flow_gt.detach(), str(_id))
        _id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output_path', help="output viz")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--raft', help="checkpoint from the RAFT paper?", type=bool, default=True)
    parser.add_argument('--fcbam', help='Add CBAM after the feature network?', type=bool, default=False)
    parser.add_argument('--ccbam', help='Add CBAM after the context network?', type=bool, default=False)
    parser.add_argument('--deform', help='Add deformable convolution?', type=bool, default=False)
    parser.add_argument('--attack_type', help='Attack type options: None, FGSM, PGD', type=str, default='FGSM')
    parser.add_argument('--epsilon', help='epsilon?', type=int, default=10.0)
    parser.add_argument('--channel', help='Color channel options: 0, 1, 2, -1 (all)', type=int, default=-1)  
    parser.add_argument('--iters', help='Number of iters for PGD?', type=int, default=50) 
    parser.add_argument("--pretrained", dest="pretrained", default=None, help="path to pre-trained model")
    parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="flownets")

    args = parser.parse_args()
    
    demo(args)