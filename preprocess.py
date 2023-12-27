import glob
import os
import argparse
import numpy as np
from PIL import Image
from scipy.io import loadmat

def parse_args():
    parser = argparse.ArgumentParser(description='FusionCount')
    parser.add_argument('--input_path', default='dataset', help='data input path')
    parser.add_argument('--output_path', default='data', help='data output path')
    parser.add_argument('--dataset', default='sha', help='dataset name')
    parser.add_argument('--model', default='')
    args = parser.parse_args()
    return args

def reform_data(image, points, max_size):
    wd_rate = 1
    ht_rate = 1
    if image.size[0] > image.size[1]:
        if image.size[0] >= max_size:
            wd_rate = max_size / image.size[0]
            wd = max_size
            ht = image.size[1] * wd_rate
            ht_rate = ht / image.size[1]
        else:
            wd = image.size[0]
            ht = image.size[1]
    elif image.size[1] > image.size[0]:
        if image.size[1] >= max_size:
            ht_rate = max_size / image.size[1]
            ht = max_size
            wd = image.size[0] * ht_rate
            wd_rate = wd / image.size[0]
        else:
            wd = image.size[0]
            ht = image.size[1]
    else:
        if image.size[0] >= max_size:
            wd = max_size
            ht = max_size
            wd_rate = max_size / image.size[0]
            ht_rate = max_size / image.size[1]
        else:
            wd = image.size[0]
            ht = image.size[1]
    resized_image = image.resize((int(wd), int(ht)))
    resized_points = [[int(x * wd_rate), int(y * ht_rate)] for x,y in points]
    return resized_image, resized_points


def preprocess_nwpu(input_path, output_path, dataset, max_size):
    images_by_mode = dict()
    tests = []
    with open(os.path.join(input_path, 'val.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            image_name = line.strip().split(' ')[0] + '.jpg'
            tests.append(image_name)
    images = sorted(glob.glob(os.path.join(input_path, 'images', '*.jpg')))[:3609]
    train_images = []
    test_images = []
    for image in images:
        if os.path.basename(image) not in tests:
            train_images.append(image)
        else:
            test_images.append(image)
    images_by_mode['train'] = train_images[:int(len(train_images) * 0.9)]
    images_by_mode['val'] = train_images[int(len(train_images) * 0.9):]
    images_by_mode['test'] = test_images
    for mode, image_paths in images_by_mode.items():
        save_dir = os.path.join(output_path, mode)
        os.makedirs(save_dir, exist_ok=True)
        count = 1
        for image_path in image_paths:
            name = f'{dataset}_{count:05d}.jpg'
            image_save_path = os.path.join(save_dir, name)
            label_path = image_path.replace('.jpg', '.mat').replace('images', 'mats')
            points = loadmat(label_path)['annPoints'].astype(np.float32)
            if not points.any():
                continue
            image = Image.open(image_path).convert('RGB')
            image, points = reform_data(image, points, max_size)
            image.save(image_save_path)
            label_save_path = image_save_path.replace('jpg', 'npy')
            np.save(label_save_path, points)
            count += 1


def preprocess_qnrf(input_path, output_path, dataset, max_size):
    images_by_mode = dict()
    train_images = glob.glob(os.path.join(input_path, 'Train', '*.jpg'))
    test_images = glob.glob(os.path.join(input_path, 'Test', '*.jpg'))
    images_by_mode['train'] = train_images[:int(len(train_images)*0.9)]
    images_by_mode['val'] = train_images[int(len(train_images)*0.9):]
    images_by_mode['test'] = test_images
    for mode, image_paths in images_by_mode.items():
        save_dir = os.path.join(output_path, mode)
        os.makedirs(save_dir, exist_ok=True)
        count = 1
        for image_path in image_paths:
            name = f'{dataset}_{count:05d}.jpg'
            image_save_path = os.path.join(save_dir, name)
            label_path = image_path.replace('.jpg', '_ann.mat')
            points = loadmat(label_path)['annPoints'].astype(np.float32)
            if not points.any():
                continue
            image = Image.open(image_path).convert('RGB')
            image, points = reform_data(image, points, max_size)
            image.save(image_save_path)
            label_save_path = image_save_path.replace('jpg', 'npy')
            np.save(label_save_path, points)
            count += 1


def preprocess_sh(input_path, output_path, dataset, max_size):
    images_by_mode = dict()
    train_images = glob.glob(os.path.join(input_path, 'train_data', 'images', '*.jpg'))
    test_images = glob.glob(os.path.join(input_path, 'test_data', 'images', '*.jpg'))
    images_by_mode['train'] = train_images[:int(len(train_images) * 0.9)]
    images_by_mode['val'] = train_images[int(len(train_images) * 0.9):]
    images_by_mode['test'] = test_images
    for mode, image_paths in images_by_mode.items():
        save_dir = os.path.join(output_path, mode)
        os.makedirs(save_dir, exist_ok=True)
        count = 1
        for image_path in image_paths:
            name = f'{dataset}_{count:05d}.jpg'
            image_save_path = os.path.join(save_dir, name)
            label_path = image_path.replace('.jpg', '.mat').replace('IMG', 'GT_IMG').replace('images', 'ground-truth')
            points = loadmat(label_path)['image_info'][0][0][0][0][0].astype(np.float32)
            if not points.any():
                continue
            image = Image.open(image_path).convert('RGB')
            image, points = reform_data(image, points, max_size)
            image.save(image_save_path)
            label_save_path = image_save_path.replace('jpg', 'npy')
            np.save(label_save_path, points)
            count += 1

def run(args):
    if args.dataset == 'all':
        print(f'preprocessing sha')
        preprocess_sh(os.path.join(args.input_path, 'sha'), os.path.join(args.output_path, 'sha'), 'sha', 1920)
        print(f'preprocessing shb')
        preprocess_sh(os.path.join(args.input_path, 'shb'), os.path.join(args.output_path, 'shb'), 'shb', 1920)
        print(f'preprocessing qnrf')
        preprocess_qnrf(os.path.join(args.input_path, 'qnrf'), os.path.join(args.output_path, 'qnrf'), 'qnrf', 1920)
        print(f'preprocessing nwpu')
        preprocess_nwpu(os.path.join(args.input_path, 'nwpu'), os.path.join(args.output_path, 'nwpu'), 'nwpu', 1920)
    else:
        for dataset_name in args.dataset.split('_'):
            print(f'preprocessing {dataset_name}')
            if dataset_name == 'sha' or dataset_name == 'shb':
                preprocess_sh(os.path.join(args.input_path, dataset_name), output_path, dataset_name, 1920)
            elif dataset_name == 'qnrf':
                preprocess_qnrf(os.path.join(args.input_path, dataset_name), output_path, dataset_name, 1920)
            elif dataset_name == 'nwpu':
                preprocess_nwpu(os.path.join(args.input_path,dataset_name), output_path, dataset_name, 1920)


if __name__ == '__main__':
    args = parse_args()
    args.output_path = os.path.join(args.model, args.output_path)
    input_path = os.path.join(args.input_path, args.dataset)
    output_path = os.path.join(args.output_path, args.dataset)
    if args.dataset_name == 'sha' or args.dataset_name == 'shb':
        preprocess_sh(input_path, output_path, args.dataset_name, 1920)
    elif args.dataset_name == 'qnrf':
        preprocess_qnrf(input_path, output_path, args.dataset_name, 1920)
    elif args.dataset_name == 'nwpu':
        preprocess_nwpu(input_path, output_path, args.dataset_name, 1920)


















