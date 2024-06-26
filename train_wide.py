import argparse
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
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
        self.alpha_p = args.alpha_p
        self.alpha_n = args.alpha_n
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
        # self.optimizer = optim.AdamW(self.transformer.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
        
        self.best = 0
        self.step = 0
        if not args.eval:
            self.writer = SummaryWriter(args.log_dir)
            self.weight_path = Path(args.log_dir).joinpath("weight")
            self.weight_path.mkdir(parents=True, exist_ok=True)
        # os.makedirs(self.weight_path, exist_ok=True)
        
    def load_weight(self, path):
        assert Path(path).exists()
        self.transformer.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded transformer weight at {path}")
    
    @staticmethod
    def extract_pn(similarity: torch.Tensor):
        '''
        A batch of data contains #batch ground/satellite images
        Given matrix M[i, j] = similarity(G[i], S[j])
        M: [batch, batch]
        G: [batch, dim]
        S: [batch, dim]
        i-th ground image corresponds to i-th satellite image (diagonal)
        Therefore, diagonal elements are positive, while others are negative
        '''
        # Create a mask for diagonal elements
        diag_mask = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        # Invert the mask to get non-diagonal elements
        non_diag_mask = ~diag_mask
        # Extract elements using the mask
        positive = similarity[diag_mask]
        negative = similarity[non_diag_mask]
        return positive, negative
    
    @staticmethod
    def similarity(grd, sat):
        '''
        grd: [batch, dim]
        sat: [batch, dim]
        '''
        # diff = grd[:, None, :] - sat[None, :, :]
        # matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        matrix = F.cosine_similarity(grd[:, None, :], sat[None, :, :], dim=-1)
        return matrix
    
    def extract_feature(self, grd_img, sat_img):
        _, sat_map = self.sat_backbone.extract_features_multiscale(2 * (sat_img / 255.0) - 1.0)
        _, grd_map = self.grd_backbone.extract_features_multiscale(2 * (grd_img / 255.0) - 1.0)
        # No need to calculate gradients for trained backbone
        sat_feature, grd_feature = self.transformer(sat_map[15].detach(), grd_map[15].detach())
        return sat_feature, grd_feature
    
    @torch.no_grad()
    def eval_epoch(self, val_loader, topn=5):
        self.transformer.eval()
        record = []
        for data in tqdm(val_loader, desc="Eval", leave=False, ncols=140):
            grd_img, sat_img, *_ = data
            batch_size = grd_img.size(0)
            
            sat_feature, grd_feature = self.extract_feature(grd_img.to(self.device), sat_img.to(self.device))
            matrix = self.similarity(grd_feature, sat_feature)
            
            # Distance      The lower the more possible
            # topn_index = torch.topk(matrix, topn, dim=-1, largest=False).indices
            
            # Similarity    The higher the more possible
            topn_index = torch.topk(matrix, topn, dim=-1).indices
            # i-th row (ground image) corresponds to i-th col (satellite image)
            index = torch.arange(batch_size, device=matrix.device).reshape(batch_size, -1)
            accuracy = (index == topn_index).any(dim=-1).sum() / batch_size
            record.append(accuracy.item())
            
        return np.mean(record)
    
    def loss_fn(self, matrix):
        '''
        matrix: [batch, batch]
        Distance or Similarity
        '''
        positive, negative = self.extract_pn(matrix)
        p_term = torch.mean(torch.exp(positive))
        n_term = torch.mean(torch.exp(negative))
        loss = -torch.log(1 + p_term / n_term)
        # p_term = torch.mean(torch.log(1 + torch.exp(-self.alpha_p * (positive - 0.7)))) / self.alpha_p
        # n_term = torch.mean(torch.log(1 + torch.exp(self.alpha_n * (negative - 0)))) / self.alpha_n
        # loss = p_term + n_term
        return loss
    
    def train_epoch(self, train_loader):
        self.transformer.train()
        for data in (pbar := tqdm(train_loader, desc="Train", leave=False, ncols=140)):
            self.step += 1
            grd_img, sat_img, *_ = data
            
            sat_feature, grd_feature = self.extract_feature(grd_img.to(self.device), sat_img.to(self.device))
            matrix = self.similarity(grd_feature, sat_feature)
            
            self.optimizer.zero_grad()
            loss = self.loss_fn(matrix)
            loss.backward()
            self.optimizer.step()
            
            if self.step % 25 == 0:
                pbar.set_postfix_str(f"loss: {loss.item():.3f}")
            self.writer.add_scalar("Train/loss", loss.item(), self.step)
        # self.scheduler.step()
    
    def train(self, args):
        train_dataset, val_dataset = fetch_dataset(args, "train", args.dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        for epoch in (pbar := tqdm(range(args.epochs), desc="Epochs", leave=True, ncols=140)):
            self.train_epoch(train_loader)
            accuracy = self.eval_epoch(val_loader, args.topn)
            
            torch.save(self.transformer.state_dict(), self.weight_path.joinpath(f"{epoch:03d}.pt"))
            if accuracy > self.best:
                self.best = accuracy
                torch.save(self.transformer.state_dict(), self.weight_path.joinpath("best.pt"))
            
            self.writer.add_scalar("Eval/Accuracy", accuracy, epoch + 1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}, best: {self.best:3f}")
        

if __name__ == "__main__":
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--eval',       default=False, action='store_true', help="evaluate/train")
    parser.add_argument('--weight',     type=str, default=root.joinpath("weight/Siamese.pt"), help="transformer weight")
    parser.add_argument('--device',     type=str, default="cuda:2", help="GPU")
    # parser.add_argument('--config',     type=str, default=root.joinpath("HC_Net/models/config/VIGOR/train-vigor.json"), help="path of config file")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--conv_k',     type=int, default=3, help="convolution output kernel")
    parser.add_argument('--conv_c',     type=int, default=256, help="convolution output channel")
    parser.add_argument('--dim',        type=int, default=1024, help="feature dim")
    parser.add_argument('--shared_w',   type=bool, default=False, help="use shared weight for grd/sat")
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--wdecay',     type=float, default=5e-4)
    parser.add_argument('--epsilon',    type=float, default=1e-7)
    parser.add_argument('--topn',       type=int, default=5)
    parser.add_argument('--alpha_p',    type=float, default=5)
    parser.add_argument('--alpha_n',    type=float, default=20)
    parser.add_argument('--ckpt',       type=str, default=root.joinpath("best_checkpoint_same.pth"), help="HC-Net weight")
    parser.add_argument('--dataset',    type=str, default=root.joinpath("HC_Net/Data/VIGOR"), help='dataset')    
    parser.add_argument('--log-dir',    type=str, default="log/test", help="tensorboard/weight location")
    
    # parser.add_argument('--model', default=None,help="restore model") 
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--ori_noise', type=float, default=45.0, help='orientation noise for VIGOR')

    parser.add_argument('--lev0', default=True, action='store_true', help='warp no')
    parser.add_argument('--flow', default=True, action='store_true', help='GMA input shape')
    parser.add_argument('--augment', default=False, action='store_true', help='Use albumentations to augment data')
    parser.add_argument('--orien', default=False, action='store_true', help='Add orientation loss')
    parser.add_argument('--p_siamese', default=True, action='store_true', help='Use siamese or pseudo-siamese backbone')
    parser.add_argument('--cross_area', default=False, action='store_true', help='Cross_area or same_area')
    parser.add_argument('--CNN16', default=True, action='store_true', help='Feature map size')
    parser.add_argument('--orig_label', default=False, action='store_true', help='Choose label for VIGOR')

    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--sat_size', type=int, default=640)
    parser.add_argument('--zoom', type=int, default=20)
    
    args = parser.parse_args()
    
    experiment = Experiment(args)
    
    if args.eval:
        experiment.load_weight(args.weight)
        test_loader = fetch_dataset(args, "eval", args.dataset)
        accuracy = experiment.eval_epoch(test_loader, args.topn)
        print(f"Accuracy: {accuracy}")
    else:
        experiment.train(args)
    