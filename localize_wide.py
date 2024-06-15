import cv2
import copy
import argparse
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from HC_Net.models.network import HCNet
from HC_Net.models.utils.utils import show_overlap
from HC_Net.models.utils.torch_geometry import get_perspective_transform
from utility import load_model
from model_wide import Transformer
from dataset_wide import fetch_dataset

class APP:
    def __init__(self, args):
        self.device = args.device
        self.HC_Net = HCNet(args)
        load_model(args.ckpt, self.HC_Net)
        self.HC_Net.to(self.device)
        
        self.sat_backbone = self.HC_Net.sat_efficientnet
        self.grd_backbone = self.HC_Net.grd_efficientnet
        
        self.transformer = Transformer(args).to(self.device)
        self.load_weight(args.weight)
        
        self.topn = args.topn
        self.iters_lev0 = args.iters_lev0
        
    def load_weight(self, path):
        assert Path(path).exists()
        self.transformer.load_state_dict(torch.load(path, map_location=self.device))
        self.transformer.eval()
        print(f"Loaded transformer weight at {path}")
    
    @torch.no_grad()
    def extract_feature(self, grd_img, sat_img):
        grd_img = 2 * (grd_img / 255.0) - 1.0
        sat_img = 2 * (sat_img / 255.0) - 1.0
        _, sat_map = self.sat_backbone.extract_features_multiscale(sat_img)
        _, grd_map = self.grd_backbone.extract_features_multiscale(grd_img)
        # No need to calculate gradients for trained backbone
        sat_feature, grd_feature = self.transformer(sat_map[15].detach(), grd_map[15].detach())
        return sat_feature, grd_feature
    
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
    
    @staticmethod
    def get_homography(displacement, W, H):
        four_point_org = torch.zeros((2, 2, 2)).to(displacement.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([W-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, H-1])
        four_point_org[:, 1, 1] = torch.Tensor([W-1, H-1])
        four_point_org = four_point_org.repeat(displacement.shape[0], 1, 1, 1)
        four_point_new = four_point_org + displacement
        H = get_perspective_transform(four_point_org.flatten(2).permute(0,2,1), four_point_new.flatten(2).permute(0,2,1))
        H = H.detach().cpu().numpy()
        return H
    
    @staticmethod
    def get_similarity(G, homography, S):
        C, H, W = G.shape
        
        # focus on part of image (less distorted)
        start, end = H // 5, H // 5 * 4
        temp = copy.deepcopy(G[:, start:end, start:end])
        G[:, :, :] = 0
        G[:, start:end, start:end] = temp
        
        # perspective transformation
        grd = cv2.warpPerspective(G.permute(1, 2, 0).detach().cpu().numpy(), homography, (W, H))
        sat = S.permute(1, 2, 0).detach().cpu().numpy()
        mask = (cv2.cvtColor(grd, cv2.COLOR_RGB2GRAY) != 0)
        
        # cosine similarity
        grd_mask, sat_mask = grd[mask].flatten(), sat[mask].flatten()
        norm = np.linalg.norm(grd_mask) * np.linalg.norm(sat_mask)
        cosine = np.dot(grd_mask, sat_mask) / norm
        return cosine, grd
    
    @torch.no_grad()    
    def localize(self, grd_img, sat_img, sat_gps):
        N, C, H, W = grd_img.shape
        sat_feature, grd_feature = self.extract_feature(grd_img, sat_img)
        matrix = self.similarity(grd_feature, sat_feature)
        topn_value, topn_index = torch.topk(matrix, self.topn, dim=-1)
        # only use first query
        maximum, argmax, final_img = 0, 0, None
        for index, confidence in zip(topn_index[0], topn_value[0]):
            G, S, GPS = grd_img[0].unsqueeze(0), sat_img[index].unsqueeze(0), sat_gps[index].unsqueeze(0)
            # displacement = self.HC_Net(G, S, sat_gps=GPS, iters_lev0=self.iters_lev0, test_mode=True)
            # homography = self.get_homography(displacement, H, W)
            displacement, corr_fn = self.HC_Net(G, S, sat_gps=GPS, iters_lev0=self.iters_lev0, test_mode=False)
            h, w = corr_fn.shape[-2:]
            corr_map = corr_fn.view(1, h, w, h, w)[:, h//2, w//2, :, :]
            corr = torch.max(corr_map)
            
            homography = self.get_homography(displacement[-1], H, W)
            s, grd = self.get_similarity(G[0], homography[0], S[0])
            # result = show_overlap(grd, sat, H[0])
            
            # error = np.mean(np.square(grd[mask] - sat[mask]))
            metric = confidence * s * corr
            # print(f"{index:2d}: {confidence:.4f} {corr:.4f} {s * corr:.4f} {confidence * s * corr:.4f}")
            if metric > maximum:
                maximum = metric
                argmax = index
                final_img = grd
                
        # print(argmax.item())
        return torch.from_numpy(final_img).to(self.device).permute(2, 0, 1), sat_img[argmax], argmax.item()
        
if __name__ == "__main__":
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weight',     type=str, default=root.joinpath("weight/Siamese.pt"), help="transformer weight")
    parser.add_argument('--device',     type=str, default="cuda:2", help="GPU")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--conv_k',     type=int, default=3, help="convolution output kernel")
    parser.add_argument('--conv_c',     type=int, default=256, help="convolution output channel")
    parser.add_argument('--dim',        type=int, default=1024, help="feature dim")
    parser.add_argument('--shared_w',   type=bool, default=False, help="use shared weight for grd/sat")
    parser.add_argument('--topn',       type=int, default=3)
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
    app = APP(args)
    
    loader = fetch_dataset(args, "eval", args.dataset)
    # data = next(iter(loader))
    total, correct = len(loader), 0
    for data in (pbar := tqdm(loader, leave=False, ncols=140)):
        grd_img, sat_img, grd_gps, sat_gps, *_ = [x.to(args.device) for x in data]
        G, S, argmax = app.localize(grd_img, sat_img, sat_gps)
        if argmax == 0: correct += 1
        pbar.set_postfix_str(f"Correct: {correct:3d}")
        
    print(correct / total)
    
    # print(sat_img)
    # save_image(sat_img, "database.png", nrow=8, normalize=True)
    # save_image(torch.stack([S, G, grd_img[0].to(args.device)]), "result.png", nrow=3, normalize=True)