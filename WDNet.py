import torch, time, os, pickle, time
import pathlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from generator import generator
from discriminator import discriminator
from tensorboardX import SummaryWriter
from vgg import Vgg16
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from PIL import Image


class WDNet(object):
    def __init__(
        self,
        train_data_loader,
        val_data_loader,
        test_data_loader,
        epochs=20,
        epoch_offset=1,
        root_save_dir='./output',
        log_dir='./log',
        gpu_mode=True,
        lr=0.000002,
        beta1=0.5,
        beta2=0.999,
        log_interval_second=10,
        eval_samples_to_save=50,
        pretrained_input=False,
    ):
        self.epochs = epochs
        self.epoch_offset = epoch_offset
        self.log_dir = log_dir
        self.gpu_mode = gpu_mode
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.last_log_time = time.time()
        self.log_interval_second = log_interval_second
        self.eval_samples_to_save = eval_samples_to_save
        self.root_save_dir = root_save_dir

        self.g = generator(3)
        self.d = discriminator(input_dim=6)
        if epoch_offset > 1:
            print(f"loading model from epoch {epoch_offset-1}")
            self.load(epoch_offset-1)
        self.vgg = Vgg16()
        self.g_optimizer = optim.Adam(self.g.parameters(), lr=lr, betas=(beta1, beta2))
        self.d_optimizer = optim.Adam(self.d.parameters(), lr=lr, betas=(beta1, beta2))
        
        if self.gpu_mode:
            self.g.cuda()
            self.d.cuda()
            self.vgg.cuda()
            self.bce = nn.BCEWithLogitsLoss().cuda()
            self.l1 = nn.L1Loss().cuda()
            self.mse = nn.MSELoss().cuda()
        else:
            self.bce = nn.BCEWithLogitsLoss()
            self.l1 = nn.L1Loss()
            self.mse = nn.MSELoss()

        self.grad_scaler_generator = GradScaler()
        self.grad_scaler_discriminator = GradScaler()

        def weight_init(m):
            classname = m.__class__.__name__
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        if epoch_offset == 1:
            self.g.apply(weight_init)
            self.d.apply(weight_init)
        elif epoch_offset < 1:
            raise Exception("epoch_offset < 1")

    def do_logging(self):
        now = time.time()
        if now - self.last_log_time < self.log_interval_second:
            return False
        self.last_log_time = now
        return True

    def train(self):
        print('training start!')
        self.vgg.eval()
        writer = SummaryWriter(log_dir='log')
        dataloader_length = len(self.train_data_loader)

        for epoch in range(self.epochs):
            self.d.train()
            self.g.train()
            epoch += self.epoch_offset
            offset_point = epoch*dataloader_length
            print(f"start training epoch {epoch}")

            for i, (x_, y_, mask, balance, alpha, w) in enumerate(tqdm(self.train_data_loader)):
                write_log = self.do_logging()
                i_point = offset_point + i
                if self.gpu_mode:
                    x_, y_, mask, balance, alpha, w = x_.cuda(), y_.cuda(), mask.cuda(), balance.cuda(), alpha.cuda(), w.cuda()

                self.g_optimizer.zero_grad()
                with autocast():
                    g_, g_mask, g_alpha, g_w, i_watermark = self.g(x_)
                    d_fake = self.d(x_, g_)
                    g_loss = self.bce(d_fake, torch.ones_like(d_fake))
                    with torch.no_grad():
                        feature_g = self.vgg(g_)
                        feature_real = self.vgg(y_)
                    vgg_loss = 0.0

                    for j in range (3):
                        vgg_loss += self.mse(feature_g[j], feature_real[j])

                    balance_weight = balance.size(0)*balance.size(1)*balance.size(2)*balance.size(3)/balance.sum()
                    mask_weight = mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()

                    mask_loss = balance_weight*self.l1(g_mask*balance, mask*balance)
                    w_loss = mask_weight*self.l1(g_w*mask, w*mask)
                    alpha_loss = mask_weight*self.l1(g_alpha*mask, alpha*mask)
                    i_watermark_loss = mask_weight*self.l1(i_watermark*mask, y_*mask)
                    i_watermark2_loss = mask_weight*self.l1(g_*mask, y_*mask)

                    g_loss_all = g_loss + 10.0*mask_loss + 10.0*w_loss + 10.0*alpha_loss + 50.0*(0.7*i_watermark2_loss + 0.3*i_watermark_loss) + 0.01*vgg_loss

                self.grad_scaler_generator.scale(g_loss_all).backward()
                self.grad_scaler_generator.step(self.g_optimizer)
                self.grad_scaler_generator.update()

                if ((i+1)%3) == 0 :
                    self.d_optimizer.zero_grad()

                    with autocast():
                        d_real = self.d(x_, y_)
                        d_real_loss = self.bce(d_real, torch.ones_like(d_real))

                        g_ = g_.detach()

                        d_fake = self.d(x_, g_)
                        d_fake_loss = self.bce(d_fake, torch.zeros_like(d_fake))

                        d_loss = (d_real_loss + d_fake_loss)/2

                    self.grad_scaler_discriminator.scale(d_loss).backward()
                    self.grad_scaler_discriminator.step(self.d_optimizer)
                    self.grad_scaler_discriminator.update()
                    if write_log:
                        writer.add_scalar('gans/d_loss', d_loss.data, i_point)

                if write_log:
                    writer.add_scalar('gans/g_loss', g_loss.data, i_point)
                    writer.add_scalar('gans/g_loss_all', g_loss_all.data, i_point)
                    writer.add_scalar('image/w_loss', w_loss, i_point)
                    writer.add_scalar('image/alpha_loss', alpha_loss, i_point)
                    writer.add_scalar('image/mask_loss', mask_loss, i_point)
                    writer.add_scalar('image/i_watermark_loss', i_watermark_loss, i_point)
                    writer.add_scalar('image/i_watermark2_loss', i_watermark2_loss, i_point)
                    writer.add_scalar('image/vgg_loss', vgg_loss, i_point)

                if ((i+1)%200) == 0:
                    self.save_sample('train', epoch, i, (y_, mask, alpha, w, x_), (g_, g_mask, g_alpha, g_w, i_watermark))

            print(f"done training for epoch {epoch}, saving model...")
            self.save(epoch)

            print(f"start evaluate for epoch {epoch}")
            self.g.eval()
            self.d.eval()
            g_loss_arr = list()
            g_loss_all_arr = list()
            w_loss_arr = list()
            alpha_loss_arr = list()
            mask_loss_arr = list()
            i_watermark_loss_arr = list()
            i_watermark2_loss_arr = list()
            vgg_loss_arr = list()
            for i, (x_, y_, mask, balance, alpha, w) in enumerate(tqdm(self.val_data_loader)):
                if self.gpu_mode:
                    x_, y_, mask, balance, alpha, w = x_.cuda(), y_.cuda(), mask.cuda(), balance.cuda(), alpha.cuda(), w.cuda()
                with torch.no_grad():
                    g_, g_mask, g_alpha, g_w, i_watermark = self.g(x_)
                    d_fake = self.d(x_, g_)
                    g_loss = self.bce(d_fake, torch.ones_like(d_fake))
                    feature_g = self.vgg(g_)
                    feature_real = self.vgg(y_)

                    vgg_loss = 0.0
                    for j in range (3):
                        vgg_loss += self.mse(feature_g[j], feature_real[j])

                    balance_weight = balance.size(0)*balance.size(1)*balance.size(2)*balance.size(3)/balance.sum()
                    mask_weight = mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()

                    mask_loss = balance_weight*self.l1(g_mask*balance, mask*balance)
                    w_loss = mask_weight*self.l1(g_w*mask, w*mask)
                    alpha_loss = mask_weight*self.l1(g_alpha*mask, alpha*mask)
                    i_watermark_loss = mask_weight*self.l1(i_watermark*mask, y_*mask)
                    i_watermark2_loss = mask_weight*self.l1(g_*mask, y_*mask)

                    g_loss_all = g_loss + 10.0*mask_loss + 10.0*w_loss + 10.0*alpha_loss + 50.0*(0.7*i_watermark2_loss + 0.3*i_watermark_loss) + 0.01*vgg_loss

                    g_loss_arr.append(g_loss)
                    g_loss_all_arr.append(g_loss_all)
                    w_loss_arr.append(w_loss)
                    alpha_loss_arr.append(alpha_loss)
                    mask_loss_arr.append(mask_loss)
                    i_watermark_loss_arr.append(i_watermark_loss)
                    i_watermark2_loss_arr.append(i_watermark2_loss)
                    vgg_loss_arr.append(vgg_loss)

                if i < self.eval_samples_to_save:
                    self.save_sample('eval', epoch, i, (y_, mask, alpha, w, x_), (g_, g_mask, g_alpha, g_w, i_watermark))

            samples = len(self.val_data_loader)
            writer.add_scalar('eval/g_loss', sum(g_loss_arr)/samples, epoch)
            writer.add_scalar('eval/g_loss_all', sum(g_loss_all_arr)/samples, epoch)
            writer.add_scalar('eval/w_loss', sum(w_loss_arr)/samples, epoch)
            writer.add_scalar('eval/alpha_loss', sum(alpha_loss_arr)/samples, epoch)
            writer.add_scalar('eval/mask_loss', sum(mask_loss_arr)/samples, epoch)
            writer.add_scalar('eval/i_watermark_loss', sum(i_watermark_loss_arr)/samples, epoch)
            writer.add_scalar('eval/i_watermark2_loss', sum(i_watermark2_loss_arr)/samples, epoch)
            writer.add_scalar('eval/vgg_loss', sum(vgg_loss_arr)/samples, epoch)

            print(f"done evaluation for epoch {epoch}")
            print(f"generating test samples for epoch {epoch}")

            self.save_test_images('test', epoch)

            print(f"done generating test samples for epoch {epoch}")
            print('='*50)

    def save_test_images(self, folder, epoch):
        all_output = list()
        for x in tqdm(self.test_data_loader):
            if self.gpu_mode:
                x = x.cuda()
            with torch.no_grad():
                g_, g_mask, g_alpha, g_w, i_watermark = self.g(x)

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

        fmt = 'epoch_%d'
        folder = os.path.join(self.root_save_dir, folder, fmt%epoch)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        filename_fmt = '%d.jpg'
        for i, image in enumerate(tqdm(all_output)):
            image = Image.fromarray(image)
            save_path = os.path.join(folder, filename_fmt%i)
            image.save(save_path)

    def save_sample(self, folder, epoch, idx, x, y):
        fmt = 'epoch_%d'
        folder = os.path.join(self.root_save_dir, folder, fmt%epoch)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        filename_fmt = '%d.jpg'
        x_arr = list()
        y_arr = list()
        for i in range(len(x)):
            x_arr.append(x[i][0])
            y_arr.append(y[i][0])

        items = list()
        for i in range(len(x_arr)):
            x_item = x_arr[i].detach().cpu().numpy()
            y_item = y_arr[i].detach().cpu().numpy()
            x_item = np.transpose(x_item, (1,2,0))
            y_item = np.transpose(y_item, (1,2,0))

            if x_item.shape[2] == 1:
                x_item = np.repeat(x_item, 3, axis=2)
            if y_item.shape[2] == 1:
                y_item = np.repeat(y_item, 3, axis=2)

            item = np.concatenate([x_item, y_item], axis=0)
            items.append(item)
        final = np.concatenate(items, axis=1)
        final *= 255.0
        final = final.astype(np.uint8)
        image = Image.fromarray(final)
        image.save(os.path.join(folder, filename_fmt%idx))

    def save(self, epoch):
        fmt = 'epoch_%d'
        folder = os.path.join(self.root_save_dir, 'models', fmt%epoch)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        torch.save(self.g.state_dict(), os.path.join(folder, 'g.pkl'))
        torch.save(self.d.state_dict(), os.path.join(folder, 'd.pkl'))

    def load(self, epoch):
        fmt = 'epoch_%d'
        folder = os.path.join(self.root_save_dir, 'models', fmt%epoch)
        self.g.load_state_dict(torch.load(os.path.join(folder, 'g.pkl')))
        self.d.load_state_dict(torch.load(os.path.join(folder, 'd.pkl')))
