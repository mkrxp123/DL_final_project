import torch
from torchvision import models
from torch import nn

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # mhsa = [nn.MultiheadAttention(128, 8, batch_first=True) for i in range(4)]
        # self.MHSA = nn.Sequential(*mhsa)
        
        self.MHSA = nn.MultiheadAttention(320, 8, batch_first=True)
        encoder_layers = nn.TransformerEncoderLayer(d_model=320, nhead=8, dim_feedforward=2048, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)

        self.linear = nn.Linear(320 * 16 * 16, 512)
        
    def forward(self, sat_map: torch.Tensor, grd_map: torch.Tensor):
        '''
        two inputs are encoded by 
        sat_map: [batch, 320, 16, 16]
        grd_map: [batch, 320, 16, 16]
        '''
        batch_size, c, h, w = sat_map.shape
        sat_map = sat_map.reshape(batch_size, c, -1).permute(2, 0, 1)
        grd_map = grd_map.reshape(batch_size, c, -1).permute(2, 0, 1)
        
        # sat_map, _ = self.MHSA(sat_map, sat_map, sat_map)
        # grd_map, _ = self.MHSA(grd_map, grd_map, grd_map)
        sat_map = self.transformer_encoder(sat_map)
        grd_map = self.transformer_encoder(grd_map)
        print(sat_map)
        sat_map = self.linear(sat_map.reshape(batch_size, -1))
        grd_map = self.linear(grd_map.reshape(batch_size, -1))
        
        return sat_map, grd_map
    
if __name__ == "__main__":
    device = torch.device("cuda:0")
    transformer = Transformer().to(device)
    # print(transformer)
    input = torch.randn((16, 320, 16, 16), device=device)
    s, g = transformer(input, input)
    # print(torch.cuda.memory_summary(device=device, abbreviated=False))