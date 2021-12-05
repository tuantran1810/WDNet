from PIL import Image
import numpy as np
import os.path as osp
import io
import os
import sys
import pickle
import matplotlib.pyplot as plt

image_size = 256
root_path = '/media/tuantran/rapid-data/chotot_watermark_removal_train'


def get_all_data_paths(path):
	lst = list()
	for file_name in os.listdir(path):
		segments = file_name.split('.')
		if segments[1] != 'pkl':
			continue
		lst.append(osp.join(path, file_name))
	return lst

def load_data_file(path):
    data = None
    with open(path, 'rb') as fd:
        data = pickle.load(fd)

    alpha = data['alpha']
    mask = np.unpackbits(data['mask']).reshape(image_size,image_size)
    balanced_mask = np.unpackbits(data['balanced_mask']).reshape(image_size,image_size)

    mask = np.uint8(mask*255)
    alpha_mask = np.uint8(mask*alpha)
    balanced_mask = np.uint8(balanced_mask*255)

    mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)
    alpha_mask = np.repeat(np.expand_dims(alpha_mask, axis=2), 3, axis=2)
    balanced_mask = np.repeat(np.expand_dims(balanced_mask, axis=2), 3, axis=2)

    output = {
        'plain_image': np.array(Image.open(io.BytesIO(data['plain_image_bytes']))),
        'watermarked_image': np.array(Image.open(io.BytesIO(data['watermarked_image_bytes']))),
        'watermark_on_image': np.array(Image.open(io.BytesIO(data['watermark_on_image_bytes']))),
        'alpha': data['alpha'],
        'mask': mask,
        'alpha_mask': alpha_mask,
        'balanced_mask': balanced_mask,
    }
    return output

def main():
    train_path = osp.join(root_path, 'train')
    all_data_paths = get_all_data_paths(train_path)

    for path in all_data_paths:
        print(path)
        data = load_data_file(path)
        plain_image = data['plain_image']
        watermarked_image = data['watermarked_image']
        watermark_on_image = data['watermark_on_image']
        mask = data['mask']
        alpha_mask = data['alpha_mask']
        balanced_mask = data['balanced_mask']

        tmp1 = np.concatenate([plain_image, watermarked_image, mask], axis=0)
        tmp2 = np.concatenate([watermark_on_image, alpha_mask, balanced_mask], axis=0)
        tmp3 = np.concatenate([tmp1, tmp2], axis=1)
        plt.figure()
        plt.imshow(tmp3)
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
