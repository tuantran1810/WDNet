import random
from PIL import Image
import numpy as np
import os.path as osp
import os
import sys
import io
import multiprocessing as mp
import pickle
import pathlib
from sklearn.model_selection import train_test_split

train_samples = 20000
val_samples = 1000
test_samples = 1000
image_size = 256
root_logo = '/media/tuantran/raid-data/dataset/chotot/chotot_logo_train'
root_dataset = '/media/tuantran/raid-data/dataset/chotot/chotot_images/images'
output_path = '/media/tuantran/rapid-data/chotot_watermark_removal_small'

img_path = osp.join(root_dataset,'%s.jpg')
logo_path = osp.join(root_logo, '%s.png')
output_path_fmt = osp.join(output_path, '%s', '%s.pkl')


def get_all_image_ids():
	img_ids=list()
	for file_name in os.listdir(root_dataset):
		segments = file_name.split('.')
		if segments[1] != 'jpg':
			continue
		img_ids.append(segments[0])
	return img_ids

def get_all_logos():
	logo_images = dict()
	for file_name in os.listdir(root_logo):
		segments = file_name.split('.')
		if segments[1] != 'png':
			continue
		logo_id = segments[0]
		logo_image = Image.open(logo_path%logo_id)
		w, h = logo_image.size
		logo_image = logo_image.convert('RGBA')
		logo_images[logo_id] = logo_image
	return logo_images

img_ids = get_all_image_ids()
logo_images = get_all_logos()

def solve_balance(mask):
	height,width=mask.shape
	k = mask.sum()
	k = (int)(k)
	mask2 = (1.0-mask)*np.random.rand(height,width)
	mask2 = mask2.flatten()
	pos = np.argsort(mask2)
	balance = np.zeros(height*width, dtype=bool)
	balance[pos[:min(250*250,4*k)]] = True
	balance = balance.reshape(height,width)
	return balance

def save_image_file(idx, plain_image, watermarked_image, watermark_on_image, alpha, mask, balanced_mask, folder):
	def jpeg_byte_array(img):
		img_byte_arr = io.BytesIO()
		img.save(img_byte_arr, format='PNG')
		img_byte_arr = img_byte_arr.getvalue()
		return img_byte_arr

	plain_image_bytes = jpeg_byte_array(plain_image)
	watermarked_image_bytes = jpeg_byte_array(watermarked_image)
	watermark_on_image_bytes = jpeg_byte_array(watermark_on_image)
	data = {
		"id": idx,
		"plain_image_bytes": plain_image_bytes,
		"watermarked_image_bytes": watermarked_image_bytes,
		"watermark_on_image_bytes": watermark_on_image_bytes,
		"alpha": alpha,
		"mask": np.packbits(mask),
		"balanced_mask": np.packbits(balanced_mask),
	}

	with open(output_path_fmt%(folder, idx), 'wb') as fd:
		pickle.dump(data, fd)

def process_and_save(idx, image_id, logo_id, folder):
	image_w, image_h = image_size, image_size
	plain_image = Image.open(img_path%image_id)
	plain_image = plain_image.resize((image_w, image_h))

	alpha = random.random()*0.5 + 0.2
	logo_angle = random.randint(0,360)
	logo = logo_images[logo_id]
	logo_rotate = logo.rotate(logo_angle, expand = True)

	logo_w, logo_h = random.randint(10,image_size), random.randint(10,image_size)
	logo_rotate = logo_rotate.resize((logo_w, logo_h))

	logo_start_w = random.randint(0, image_w-logo_w)
	logo_start_h = random.randint(0, image_h-logo_h)
	logo_end_w = logo_start_w + logo_w
	logo_end_h = logo_start_h + logo_h

	logo_rotate_np = np.array(logo_rotate)
	watermarked_image = np.array(plain_image)
	logo_mask = logo_rotate_np[:,:,3:4] > (15.0/255.0)
	watermarked_image[logo_start_h:logo_end_h, logo_start_w:logo_end_w, :] = \
		watermarked_image[logo_start_h:logo_end_h, logo_start_w:logo_end_w, :]*(1.0-alpha*logo_mask) + logo_rotate_np[:,:,:3]*alpha*logo_mask
	watermarked_image = np.uint8(watermarked_image)
	
	mask = np.zeros((image_w, image_h), dtype=bool)
	mask[logo_start_h:logo_end_h, logo_start_w:logo_end_w] = logo_mask[:,:,0]

	balanced_mask = solve_balance(mask)

	logo_rotate = logo_rotate.convert('RGB')
	logo_rotate_np = np.array(logo_rotate)
	watermark_on_image = np.zeros_like(watermarked_image)
	watermark_on_image[logo_start_h:logo_end_h, logo_start_w:logo_end_w, :] = logo_rotate_np
	
	data = {
		"idx": idx,
		"plain_image": plain_image,
		"watermarked_image": Image.fromarray(watermarked_image),
		"watermark_on_image": Image.fromarray(watermark_on_image),
		"alpha": alpha,
		"mask": mask,
		"balanced_mask": balanced_mask,
		"folder": folder,
	}
	save_image_file(**data)


def wrapper(mp):
	process_and_save(**mp)

def main():
	i = 0
	tasks = list()
	sample_size = train_samples + val_samples + test_samples
	logo_list = [k for k in logo_images.keys()]
	while i < sample_size:
		for img_id in img_ids:
			if i >= sample_size:
				break
			logo_id = random.choice(logo_list)
			logo_size_ratio = random.uniform(0.2, 1)
			logo_angle = random.randint(-45,45)
			tasks.append({
				'idx': i,
				'image_id': img_id,
				'logo_id': logo_id,
			})
			i += 1

	train, tmp = train_test_split(tasks, test_size=val_samples+test_samples)
	test, val = train_test_split(tmp, test_size=test_samples)

	for i in range(len(train)):
		train[i]['folder'] = 'train'
	for i in range(len(val)):
		val[i]['folder'] = 'val'
	for i in range(len(test)):
		test[i]['folder'] = 'test'
	tasks = train + val + test

	pathlib.Path(osp.join(output_path, 'train')).mkdir(parents=True, exist_ok=True)
	pathlib.Path(osp.join(output_path, 'val')).mkdir(parents=True, exist_ok=True)
	pathlib.Path(osp.join(output_path, 'test')).mkdir(parents=True, exist_ok=True)

	pool = mp.Pool(processes=6)
	pool.map(wrapper, tasks)

if __name__ == "__main__":
	main()
