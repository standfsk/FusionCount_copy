import argparse
import torch
import os
import numpy as np
import dataset.crowd as crowd
from models import vgg19
import time
import glob
from PIL import Image
from torchvision import transforms
import cv2
from scipy.ndimage import gaussian_filter

start_time = time.time()

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--crop-size', type=int, default=512,
                    help='the crop size of the train image')
parser.add_argument('--weight_path', type=str, default='pretrained_models/model_qnrf.pth',
                    help='saved model path')
parser.add_argument('--data-path', type=str,
                    default='data/QNRF-Train-Val-Test',
                    help='saved model path')
parser.add_argument('--dataset', type=str, default='qnrf',
                    help='dataset name: qnrf, nwpu, sha, shb')
parser.add_argument('--save_path', default='result', help='final image save path')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
device = torch.device('cuda')

model_path = args.weight_path
crop_size = args.crop_size
data_path = args.data_path
#### create dataloader without gt

dataset = crowd.Crowd_no(os.path.join(data_path), crop_size, 8, method='test')

dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                         num_workers=1, pin_memory=True)
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()

for step, (inputs, name) in enumerate(dataloader):
    inputs = inputs.to(device)
    assert inputs.size(0) == 1, 'the batch size should equal to 1'
    with torch.set_grad_enabled(False):
        outputs, _ = model(inputs)
    vis_img = outputs[0, 0].cpu().numpy()
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)

    original_image = cv2.imread(os.path.join(data_path, name[0] + '.jpg'))

    vis_img.resize((original_image.shape[0], original_image.shape[1]))
    density_map = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)

    overlay_weight = 0.5
    overlay = cv2.addWeighted(original_image, 1 - overlay_weight, density_map, overlay_weight, 0)
    cv2.imwrite(os.path.join(args.save_path, name[0]+'.jpg'), overlay)