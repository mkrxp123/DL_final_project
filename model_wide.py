import torch
from torchvision import models
from torch import nn

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Conv2d(320, args.conv_c, args.conv_k)
        self.ELU = nn.ELU()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.conv_c, dim_feedforward=1024, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 4)
        
        length = 17 - args.conv_k
        self.linear = nn.Linear(args.conv_c * length * length, args.dim)
        
    def forward(self, map: torch.Tensor):
        map = self.conv(map)    # batch, 128, 14, 14
        map = self.ELU(map)
        batch_size, c, h, w = map.shape
        map = map.reshape(batch_size, c, -1).permute(0, 2, 1)
        
        map = self.encoder(map)
        map = self.linear(map.reshape(batch_size, -1))
        return map

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.grd_encoder = Encoder(args)
        self.sat_encoder = self.grd_encoder if args.shared_w else Encoder(args)
        
    def forward(self, sat_map: torch.Tensor, grd_map: torch.Tensor):
        '''
        two inputs are encoded by 
        sat_map: [batch, 320, 16, 16]
        grd_map: [batch, 320, 16, 16]
        '''
        sat_map = self.sat_encoder(sat_map)
        grd_map = self.grd_encoder(grd_map)
        return sat_map, grd_map
    
if __name__ == "__main__":
    device = torch.device("cuda:2")
    transformer = Transformer(0).to(device)
    # print(transformer)
    input = torch.randn((16, 320, 16, 16), device=device)
    s, g = transformer(input, input)