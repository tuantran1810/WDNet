import os
import torch
from torch import nn
from generator import generator
from dataset import ArrayDataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import pathlib
from torchvision import transforms

def get_all_data_paths(path):
	lst = list()
	for file_name in os.listdir(path):
		segments = file_name.split('.')
		if segments[1] != 'jpg':
			continue
		lst.append(os.path.join(path, file_name))
	return lst

def main():
    batchsize = 1
    model_path = './output/models/epoch_60/g.pkl'
    image_path = './samples/input_images/homedy'
    output_path = './samples/output_images'

    g = generator(3)
    g_tmp = torch.load(model_path)
    g.load_state_dict(g_tmp)
    g = g.cuda()
    g.eval()

    print("loading sample images")
    image_paths = get_all_data_paths(image_path)
    all_images = list()
    transformation = transforms.ToTensor()
    for image_path in image_paths:
        image = Image.open(image_path)
        w, h = image.size
        maxwh = max(w, h)
        if maxwh > 768:
            ratio = 768/maxwh
            if w > h:
                w = 768
                h = int(ratio*h)
            else:
                h = 768
                w = int(ratio*w)
        image = image.convert('RGB')
        image = image.resize((w,h))
        all_images.append(transformation(image))

    dataset = ArrayDataset(all_images, None)
    params = {
        'batch_size': batchsize,
        'shuffle': False,
        'num_workers': 1,
        'drop_last': False,
    }
    dataloader = DataLoader(dataset, **params)

    print("processing...")
    all_output = list()
    for x in tqdm(dataloader):
        x = x.cuda()
        with torch.no_grad():
            g_, g_mask, g_alpha, g_w, i_watermark = g(x)
        batchsize = g_.shape[0]
        for i in range(batchsize):
            output_images = list()
            for item in [x, g_, g_mask]:
                tmp = item[i]
                tmp *= 255.0
                tmp = tmp.detach().cpu().numpy()
                tmp = tmp.astype(np.uint8)
                if tmp.shape[0] == 1:
                    tmp = np.repeat(tmp, 3, axis=0)
                tmp = np.transpose(tmp, (1,2,0))
                output_images.append(tmp)
            output_images = np.concatenate(output_images, axis=1)
            all_output.append(output_images)

    print("done, saving images ...")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    filename_fmt = '%d.jpg'
    for i, image in enumerate(tqdm(all_output)):
        image = Image.fromarray(image)
        save_path = os.path.join(output_path, filename_fmt%i)
        image.save(save_path)

if __name__ == '__main__':
    main()
