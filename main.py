import os, torch
from WDNet import WDNet
from dataset import PathDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
from PIL import Image
import io
import cv2
from functools import partial

def get_all_data_paths(path):
	lst = list()
	for file_name in os.listdir(path):
		segments = file_name.split('.')
		if segments[1] != 'pkl':
			continue
		lst.append(os.path.join(path, file_name))
	return lst

def _process_data_file(fd, imagesize):
    data = pickle.load(fd)

    alpha = data['alpha']
    mask = np.unpackbits(data['mask']).reshape(imagesize, imagesize)
    balanced_mask = np.unpackbits(data['balanced_mask']).reshape(imagesize, imagesize)

    mask = mask*1.0
    alpha_mask = mask*alpha
    balanced_mask = balanced_mask*1.0

    mask = np.expand_dims(mask, axis=2)
    alpha_mask = np.expand_dims(alpha_mask, axis=2)
    balanced_mask = np.expand_dims(balanced_mask, axis=2)
    plain_image = np.array(Image.open(io.BytesIO(data['plain_image_bytes'])))/255.0
    watermarked_image = np.array(Image.open(io.BytesIO(data['watermarked_image_bytes'])))/255.0
    watermark_on_image = np.array(Image.open(io.BytesIO(data['watermark_on_image_bytes'])))/255.0

    mask = np.transpose(mask, (2,0,1))
    alpha_mask = np.transpose(alpha_mask, (2,0,1))
    balanced_mask = np.transpose(balanced_mask, (2,0,1))
    plain_image = np.transpose(plain_image, (2,0,1))
    watermarked_image = np.transpose(watermarked_image, (2,0,1))
    watermark_on_image = np.transpose(watermark_on_image, (2,0,1))

    output = {
        'plain_image': torch.from_numpy(plain_image).float(),
        'watermarked_image': torch.from_numpy(watermarked_image).float(),
        'watermark_on_image': torch.from_numpy(watermark_on_image).float(),
        'alpha': data['alpha'],
        'mask': torch.from_numpy(mask).float(),
        'alpha_mask': torch.from_numpy(alpha_mask).float(),
        'balanced_mask': torch.from_numpy(balanced_mask).float(),
    }
    return output

def process_data(imagesize, fd):
    out = _process_data_file(fd, imagesize)
    return (
        out['watermarked_image'],
        out['plain_image'],
        out['mask'],
        out['balanced_mask'],
        out['alpha_mask'],
        out['watermark_on_image'],
    )

def get_config():
    config = dict()

    default_root_data_path = '/media/tuantran/rapid-data/chotot_watermark_removal'
    root_data_path = os.getenv('LMR_ROOT_DATA_PATH')
    root_data_path = root_data_path if root_data_path is not None else default_root_data_path
    config['root_data_path'] = root_data_path

    default_output_path = "./output"
    output_path = os.getenv('LMR_OUTPUT_PATH')
    output_path = output_path if output_path is not None else default_output_path
    config['output_path'] = output_path

    default_log_path = "./log"
    log_path = os.getenv('LMR_LOG_PATH')
    log_path = log_path if log_path is not None else default_log_path
    config['log_path'] = log_path

    default_epochs = 20
    epochs = os.getenv('LMR_EPOCHS')
    epochs = int(epochs) if epochs is not None else default_epochs
    config['epochs'] = epochs

    default_epoch_offset = 1
    epoch_offset = os.getenv('LMR_EPOCH_OFFSET')
    epoch_offset = int(epoch_offset) if epoch_offset is not None else default_epoch_offset
    config['epoch_offset'] = epoch_offset

    default_batchsize = 3
    batchsize = os.getenv('LMR_BATCHSIZE')
    batchsize = int(batchsize) if batchsize is not None else default_batchsize
    config['batchsize'] = batchsize

    default_imagesize = 480
    imagesize = os.getenv('LMR_IMAGESIZE')
    imagesize = int(imagesize) if imagesize is not None else default_imagesize
    config['imagesize'] = imagesize

    return config

def main():
    config = get_config()
    imagesize = config['imagesize']
    root_data_path = config['root_data_path']
    batchsize = config['batchsize']

    train_lst = get_all_data_paths(os.path.join(root_data_path, 'train'))
    val_lst = get_all_data_paths(os.path.join(root_data_path, 'val'))
    train_dataset = PathDataset(train_lst, partial(process_data, imagesize))
    val_dataset = PathDataset(val_lst, partial(process_data, imagesize))
    params = {
        'batch_size': batchsize,
        'shuffle': True,
        'num_workers': 4,
        'drop_last': True,
    }
    train_dataloader, val_dataloader = DataLoader(train_dataset, **params), DataLoader(val_dataset, **params)

    net_params = {
        'epochs': config['epochs'],
        'epoch_offset': config['epoch_offset'],
        'root_save_dir': config['output_path'],
        'log_dir': config['log_path'],
    }

    gan = WDNet(
        train_dataloader,
        val_dataloader,
        **net_params,
    )

    gan.train()

if __name__ == '__main__':
    main()
