import os
import sys
sys.path.append('HC_Net')
import argparse
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from HC_Net.models.network import HCNet
from dataset_wide import fetch_dataset
from model_wide import Transformer
from utility import fetch_config, load_model


class Experiment:
    def __init__(self, args):
        self.device = args.device
        self.HC_Net = HCNet(args)

        load_model(args.ckpt, self.HC_Net)
        # checkpoint = torch.load(args.ckpt, map_location="cpu")
        # self.HC_Net.load_state_dict(checkpoint['model'])
        self.sat_backbone = self.HC_Net.sat_efficientnet.to(self.device)
        self.grd_backbone = self.HC_Net.grd_efficientnet.to(self.device)

        self.transformer = Transformer(args).to(self.device)
        self.optimizer = optim.AdamW(self.transformer.parameters(
        ), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

        self.step = 0
        self.writer = SummaryWriter(args.log_dir)
        self.weight_path = Path(args.log_dir).joinpath("weight")
        os.makedirs(self.weight_path, exist_ok=True)

    def extract_feature(self, grd_img, sat_img):
        _, sat_map = self.sat_backbone.extract_features_multiscale(sat_img)
        _, grd_map = self.grd_backbone.extract_features_multiscale(grd_img)
        # No need to calculate gradients for trained backbone
        sat_feature, grd_feature = self.transformer(
            sat_map[15].detach(), grd_map[15].detach())
        return sat_feature, grd_feature

    @torch.no_grad()
    def eval_epoch(self, val_loader, topn=5):
        # positive_distance, negative_distance, topn_index
        record = []
        index_record = []
        for data in tqdm(val_loader, desc="Eval", leave=False, ncols=140):
            grd_img, sat_img, *_ = data

            sat_feature, grd_feature = self.extract_feature(
                grd_img.to(self.device), sat_img.to(self.device))
            

            distance = grd_feature[:, None, :] - sat_feature[None, :, :]
            diag = torch.eye(distance.shape[0], dtype=torch.bool, device=distance.device)

            distance = abs(distance)

            positive_distance_mean = distance[diag].mean()
            negative_distance_mean = distance[~diag].mean()
            topn_index = torch.topk(distance, topn, dim=-1, largest=False).indices

            record.append(positive_distance_mean.item())
            index_record.append(topn_index)


            # record.append(???.item())
        return np.mean(record), index_record

    @staticmethod
    def extract_pn(distance: torch.Tensor):
        '''
        A batch of data contains #batch ground/satellite images
        Given distance D[i, j] = G[i] - S[j]
        D: [batch, batch]
        G: [batch, dim]
        S: [batch, dim]
        i-th ground image corresponds to i-th satellite image (diagonal)
        Therefore, diagonal elements are positive, while others are negative
        '''
        # Create a mask for diagonal elements
        diag_mask = torch.eye(
            distance.shape[0], dtype=torch.bool, device=distance.device)
        # Invert the mask to get non-diagonal elements
        non_diag_mask = ~diag_mask
        # Extract elements using the mask
        positive = distance[diag_mask]
        negative_distance = distance
        negative_distance[diag_mask] = 0
        negative = negative_distance.mean(dim=-1)
        # negative = distance[non_diag_mask]
        return positive, negative

    def loss_fn(self, grd, sat):
        '''
        grd: [batch, dim]
        sat: [batch, dim]
        '''
        diff = grd[:, None, :] - sat[None, :, :]

        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))

        positive, negative = self.extract_pn(dist)

        # triplet loss
        # loss = torch.clamp(positive - negative + 1, min=0).mean()
        # return loss

        ## soft margin triplet loss


    def train_epoch(self, train_loader):
        for data in tqdm(train_loader, desc="Train", leave=False, ncols=140):
            self.step += 1
            grd_img, sat_img, *_ = data
            sat_feature, grd_feature = self.extract_feature(
                grd_img.to(self.device), sat_img.to(self.device))
            # loss = ???

            self.optimizer.zero_grad()
            # loss.backward()
            self.optimizer.step()
            # self.writer.add_scalar("Train/loss", loss.item(), self.step)
        self.scheduler.step()

    def train(self, args):
        train_dataset, val_dataset = fetch_dataset(
            args, "train", root.joinpath("HC_Net/VIGOR"))
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        for epoch in (pbar := tqdm(range(args.epochs), desc="Epochs", leave=False, ncols=140)):
            self.train_epoch(train_loader)
            metric = self.eval_epoch(val_loader)

            # pbar.set_description_str(f"???: {metric}, best: {???}")
            self.writer.add_scalar("Eval/???", metric, epoch + 1)
            torch.save(self.transformer.state_dict(),
                       self.weight_path.joinpath(f"{epoch:03d}.pt"))


if __name__ == "__main__":
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()

    parser.add_argument('--device',     type=str, default="cuda", help="GPU")
    # parser.add_argument('--config',     type=str, default=root.joinpath("HC_Net/models/config/VIGOR/train-vigor.json"), help="path of config file")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--wdecay',     type=float, default=5e-4)
    parser.add_argument('--epsilon',    type=float, default=1e-7)
    parser.add_argument('--ckpt',       type=str, default=root.joinpath(
        "best_checkpoint_same.pth"), help="restore checkpoint")
    parser.add_argument('--dataset',    type=str,
                        default=root.joinpath("HC_Net/Data/VIGOR"), help='dataset')
    parser.add_argument('--log-dir',    type=str,
                        default="log", help="tensorboard/weight location")

    # parser.add_argument('--model', default=None,help="restore model")
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--mixed_precision', default=False,
                        action='store_true', help='use mixed precision')
    parser.add_argument('--ori_noise', type=float,
                        default=45.0, help='orientation noise for VIGOR')

    parser.add_argument('--lev0', default=True,
                        action='store_true', help='warp no')
    parser.add_argument('--flow', default=True,
                        action='store_true', help='GMA input shape')
    parser.add_argument('--augment', default=False, action='store_true',
                        help='Use albumentations to augment data')
    parser.add_argument('--orien', default=False,
                        action='store_true', help='Add orientation loss')
    parser.add_argument('--p_siamese', default=True, action='store_true',
                        help='Use siamese or pseudo-siamese backbone')
    parser.add_argument('--cross_area', default=False,
                        action='store_true', help='Cross_area or same_area')
    parser.add_argument('--CNN16', default=True,
                        action='store_true', help='Feature map size')
    parser.add_argument('--orig_label', default=False,
                        action='store_true', help='Choose label for VIGOR')

    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--sat_size', type=int, default=640)
    parser.add_argument('--zoom', type=int, default=20)
    parser.add_argument('--gpuid', type=list, default=[0])

    args = parser.parse_args()
    # args = fetch_config(args)
    experiment = Experiment(args)

    train_dataset, val_dataset = fetch_dataset(
        args, "train", root.joinpath("HC_Net/VIGOR"))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    experiment.eval_epoch(val_loader, topn=3)

    # experiment.train(args)
