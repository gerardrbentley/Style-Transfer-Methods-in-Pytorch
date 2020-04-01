"""
Example template for defining a system
Runs a model on a single node across N-gpus.
"""
import os
import logging as log
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import imageio

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

import utils
from pytorch_models.transformer_net import TransformerNet
from pytorch_models.vgg_extractor import Vgg16
from pytorch_models.hrnet import HRNet


class FastNeuralStyleSystem(LightningModule):
    """
    Style Transfer
    """

    def __init__(self, hparams):
        """
        Pass in parsed ArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(FastNeuralStyleSystem, self).__init__()
        self.hparams = hparams
        torch.manual_seed(hparams.seed)
        np.random.seed(hparams.seed)

        self.batch_size = hparams.batch_size
        if hparams.model == "hrnet":
            self.style_model = HRNet()
        else:
            self.style_model = TransformerNet()
        self.vgg_extractor = Vgg16(requires_grad=False)

        self.transform = transforms.Compose([
            transforms.Resize(hparams.image_size),
            transforms.CenterCrop(hparams.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        self.style_transform = transforms.Compose([
            transforms.Resize(hparams.image_size),
            transforms.CenterCrop(hparams.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        content_image = utils.load_image(
            self.hparams.content_image, scale=self.hparams.content_scale)
        self.content_image = self.style_transform(content_image)

        style = utils.load_image(os.path.join(
            'images', 'style-images', f'{hparams.style_image}.jpg'), scale=0.5)
        style = self.style_transform(style).requires_grad_(False)
        self.style_image = style.repeat(hparams.batch_size, 1, 1, 1)

        self.features_style = self.vgg_extractor(
            utils.normalize_batch(self.style_image))
        self.gram_style = [utils.gram_matrix(y) for y in self.features_style]

        # self.temp_dir = f"{self.hparams.output_dir}/{self.hparams.style_image}_steps_c_{self.hparams.content_weight}_s_{self.hparams.style_weight}"
        # os.makedirs(self.temp_dir, exist_ok=True)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        return self.style_model(x)

    def loss(self, images, stylized, inplace=True):
        stylized = utils.normalize_batch(stylized, inplace=inplace)
        images = utils.normalize_batch(images, inplace=inplace)

        features_stylized = self.vgg_extractor(stylized)
        features_images = self.vgg_extractor(images)

        content_loss = self.hparams.content_weight * \
            F.mse_loss(features_stylized.relu2_2, features_images.relu2_2)
        # style_weights = [1.0,
        #                  1.0,
        #                  1.2,
        #                  1.4,
        #                  1.4]
        style_weights = [1.0, 1.0, 1.4, 1.0, 1.0]
        style_loss = 0.
        for i, (ft_stylized, gm_s)in enumerate(zip(features_stylized, self.gram_style)):
            gm_stylized = utils.gram_matrix(ft_stylized)
            gm_s = gm_s.type_as(ft_stylized)
            c, h, w = gm_stylized.shape
            style_loss += F.mse_loss(gm_stylized, gm_s[: len(images), :, :])
            # style_loss *= style_weights[i] / (c * h * w)
        style_loss *= self.hparams.style_weight

        total_loss = content_loss + style_loss
        return total_loss, content_loss, style_loss

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, _ = batch

        y = self.forward(x)
        # print(f"input/output: {torch.cuda.current_device()}. x: {x.shape}. y: {y.shape}")
        # calculate loss
        loss_val, content_loss, style_loss = self.loss(x, y)
        # print(f"{torch.cuda.current_device()}: loss: {loss_val}. content: {content_loss}, style: {style_loss}. types: {type(loss_val)}. sizes: {loss_val.size()},dev: {loss_val.device}")
        tqdm_dict = {'train_loss': loss_val,
                     'content_loss': content_loss,
                     'style_loss': style_loss
                     }

        output = {
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

        if self.global_step % 300 == 0:
            self.stylize(self.global_step)
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the val loop
        :param batch:
        :return:
        """
        # forward pass
        x, _ = batch
        y = self.forward(x)
        # print(f"input/output: {torch.cuda.current_device()}. x: {x.shape}. y: {y.shape}")
        # calculate loss
        loss_val, content_loss, style_loss = self.loss(x, y)
        # print(f"VAL {torch.cuda.current_device()}: loss: {loss_val}. content: {content_loss}, style: {style_loss}. types: {type(loss_val)} sizes: {loss_val.size()}. dev: {loss_val.device}")
        tqdm_dict = {'val_loss': loss_val,
                     'content_loss': content_loss,
                     'style_loss': style_loss
                     }

        output = OrderedDict({
            'val_loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        if batch_idx == 0:
            torchvision.utils.save_image(y, os.path.join(
                self.hparams.output_dir, f"{self.hparams.style_image}_VAL_{self.hparams.content_weight}_{self.hparams.style_weight}.png"), scale_each=True, padding=16, normalize=True)
        # can also return just a scalar instead of a dict (return loss_val)
        return output
    # def validation_step(self, batch, batch_idx):
    #     """
    #     Lightning calls this inside the validation loop
    #     :param batch:
    #     :return:
    #     """

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return optimizer

    def prepare_data(self):
        self.train_dataset = datasets.ImageFolder(
            self.hparams.dataset, self.transform)
        if self.hparams.to_style != '':
            self.val_dataset = datasets.ImageFolder(
                self.hparams.to_style, self.style_transform)
        else:
            self.val_dataset = None

    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=4,
                          drop_last=True,
                          )

    def val_dataloader(self):
        log.info('Validation data loader called.')
        if self.val_dataset is not None:
            return DataLoader(dataset=self.val_dataset, batch_size=1, num_workers=3, drop_last=False)
        else:
            return None

    def stylize(self, id=0):
        # img_list = ['./images/content-images/gbimage2.jpeg', './images/content-images/gbimage.jpeg', './images/content-images/shirtless.jpeg', './images/content-images/little.jpeg', './images/content-images/frizimage.jpeg']
        imgs = []
        if self.val_dataset is not None:
            for i, (prepped_image, _) in enumerate(self.val_dataset):
                imgs.append(prepped_image)
        model_input = torch.stack(imgs)
        model_input = model_input.type_as(next(self.style_model.parameters()))
        # content_image = content_image.unsqueeze(0).to('cuda')

        output = self.forward(model_input)
        # utils.save_image(f"{self.hparams.output_dir}_510_{id}.png", output[0])
        torchvision.utils.save_image(output, os.path.join(
            self.temp_dir, f"{self.hparams.style_image}_{id}_grid.png"), scale_each=True, padding=16, normalize=True)

    def optimize(self):
        import os
        temp_dir = os.path.join(
            self.hparams.output_dir, f"{self.hparams.style_image}_{self.hparams.model}_w_{self.hparams.weights}_c_{self.hparams.content_weight}_s_{self.hparams.style_weight}")
        temper_dir = os.path.join(temp_dir, f"steps")
        os.makedirs(temper_dir, exist_ok=True)
        for i in range(1):
            print(torch.cuda.memory_summary(i))

        # torch.cuda.ipc_collect()
        single_opt = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            single_opt, step_size=200, gamma=0.9)

        self.content_image = self.content_image.to(
            'cuda').unsqueeze(0).requires_grad_(False)
        self.content_image = utils.normalize_batch(
            self.content_image, inplace=True)
        prediction = self.content_image.clone().requires_grad_(True)
        # prediction = torch.randn_like(self.content_image).requires_grad_(True)

        torchvision.utils.save_image(
            self.content_image, f"{temp_dir}/0_content.png", normalize=True)
        torchvision.utils.save_image(
            self.style_image, f"{temp_dir}/0_style.png", normalize=True)
        s = []
        c = []
        total = []
        saved_images = []
        for step in range(30000):
            prediction = self.forward(self.content_image)
            prediction.requires_grad_(True)

            loss_val, content_loss, style_loss = self.loss(
                self.content_image, prediction, inplace=False)
            total.append(loss_val.item())
            c.append(content_loss.item())
            s.append(style_loss.item())

            if (step+1) % 200 == 0:
                print("After %d criterions, learning:" % (step+1))
                print('Total loss: ', loss_val.item())
                print('Content loss: ', content_loss.item())
                print('Style loss: ', style_loss.item())
                print(prediction.shape)
                print(torch.unique(prediction))
            if (step+1) % 400 == 0:
                new_filename = os.path.join(
                    temper_dir, f"optim_{(step+1)}_{self.hparams.style_image}.png")
                saved_images.append(new_filename)
                torchvision.utils.save_image(
                    prediction, new_filename, normalize=True)

            single_opt.zero_grad()

            loss_val.backward()

            single_opt.step()
            # scheduler.step()
        gif_images = []
        for step_img in saved_images:
            gif_images.append(imageio.imread(step_img))
        imageio.mimsave(os.path.join(
            temp_dir, '0_optimization.gif'), gif_images)
        torchvision.utils.save_image(
            prediction, os.path.join(
                temp_dir, '0_final.png'), normalize=True)

    # def test_dataloader(self):
    #     log.info('Test data loader called.')
    #     return self.__dataloader(train=False)


def main():
    """
    Main training routine specific for this project
    :param args:
    """
    # ------------------------
    # 0 SETUP
    # ------------------------
    log.getLogger().setLevel(log.INFO)
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(
        description='Image Style Transfer Training')
    parser = pl.Trainer.add_argparse_args(parser)
    # data
    parser.add_argument('-s', '--style-image', default='cropamara', type=str)
    parser.add_argument('-d', '--dataset',
                        default=os.path.join('/', 'fridge', 'coco'), type=str)
    parser.add_argument('-t', '--to-style',
                        default=os.path.join('images', 'test'), type=str)
    parser.add_argument('-m', '--model', default='transformer', type=str)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument("--image-size", type=int, default=256,
                        help="size of training images, default is 256")

    parser.add_argument("--seed", type=int, default=4747,
                        help="random seed for training")
    parser.add_argument("--content-weight", type=float, default=1e5,
                        help="weight for content-loss, default is 1e5")
    parser.add_argument("--style-weight", type=float, default=1e10,
                        help="weight for style-loss, default is 1e10")
    parser.add_argument("--weights", type=str, default='flat',
                        help="weight for layer losses, default is 1 each")

    parser.add_argument("--content-image", type=str, default='./images/content-images/gbimage2.jpeg',
                        help="path to content image you want to stylize")
    parser.add_argument("--content-scale", type=float, default=None,
                        help="factor for scaling down the content image")
    parser.add_argument("--output-dir", type=str, default='./images/output-images/',
                        help="path for saving the output images")

    parser.add_argument("-cp", "--checkpoint", type=str, default='',
                        help="path for starting weights")
    parser.add_argument("--single", action='store_true')

    parser.set_defaults(progress_bar_refresh_rate=5,
                        gpus='0,1,2',
                        max_epochs=50,
                        overfit_pct=0.01,
                        profiler=True,
                        weights_summary='full',
                        logger=False,
                        distributed_backend="dp")
    args = parser.parse_args()

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = FastNeuralStyleSystem(args)
    if args.checkpoint is not '':
        print(f'loading checkpoint: {args.checkpoint}')
        FastNeuralStyleSystem.load_from_checkpoint(args.checkpoint)
    print(model.hparams)
    if args.single:
        print('single image optimize')
        model.to('cuda')
        model.prepare_data()
        model.optimize()
        print('Done single image')
        return
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath='./trained_models',
        save_top_k=2,
        verbose=True,
        monitor='train_loss',
        mode='min',
        prefix=args.style_image
    )
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)

    import glob
    saved_images = glob.glob(
        f"{args.output_dir}/{args.style_image}_steps_c_{args.content_weight}_s_{args.style_weight}/*png")
    gif_images = []
    for step_img in saved_images:
        gif_images.append(imageio.imread(step_img))
    imageio.mimsave(os.path.join(temp_dir, '0_optimization.gif'), gif_images)


if __name__ == '__main__':
    main()
