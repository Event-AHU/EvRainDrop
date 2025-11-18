import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models_evrd import EvRaindrop_v1

class EvRainDrop_adapted(nn.Module):
    def __init__(self, d_inp=128, d_model=256, nhead=4, nhid=128, nlayers=2, 
                 dropout=0.3, max_len=768, MAX=100, output_dim=128*768, d_static=768):
        super().__init__()
        self.d_ob = d_model // d_inp

        self.global_structure = torch.ones(d_inp, d_inp, device='cuda')

        self.raindrop = EvRaindrop_v1(
            d_inp=d_inp,
            d_model=d_model,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout,
            max_len=max_len,
            d_static=d_static,
            MAX=MAX,
            n_classes=output_dim, 
            global_structure=self.global_structure,
            sensor_wise_mask=False,
            static=True
        )

        self.raindrop.mlp_static = nn.Sequential(
            nn.Linear(d_model + 16 + d_inp, 512), 
            nn.ReLU(),
            nn.Linear(512, output_dim)
        ).cuda()

    def forward(self, x, static):
        src, times, lengths = preprocess_data(x)
        output, interacted_static, _ = self.raindrop(src, static=static, times=times, lengths=lengths)
        processed_output = postprocess_data(output, x.shape)
        return processed_output, interacted_static


def preprocess_data(data):
    data = data.float().cuda()
    batch_size = data.shape[0]
    step = 768  
    valid_step = step * 0.8
    time_mask = torch.arange(step, device=data.device) >= valid_step
    time_mask = time_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
    data = data.masked_fill(time_mask, 0.0)
    src = data.permute(2, 0, 1).contiguous()
    z_coords = data.mean(dim=1)  
    times = z_coords.permute(1, 0).contiguous()  
    lengths = torch.full((batch_size,), valid_step, dtype=torch.long, device=data.device)
    return src, times, lengths


def postprocess_data(output, original_shape):
    batch_size, d_inp, T = original_shape
    return output.view(batch_size, d_inp, T)


class EvRainDrop(nn.Module):
    def __init__(self, evrd_kwargs=None):
        super().__init__()
        default_kwargs = {
            "d_inp": 128,
            "d_model": 256,  
            "nhead": 4,
            "nhid": 128,
            "nlayers": 2,
            "dropout": 0.3,
            "max_len": 768,
            "MAX": 100,
            "output_dim": 128 * 768,
            "d_static": 768
        }
        self.evrd_kwargs = evrd_kwargs if evrd_kwargs else default_kwargs
        self.evrd_interact = EvRainDrop_adapted(**self.evrd_kwargs).cuda()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=768, 
            num_heads=12,  
            batch_first=True
        ).cuda()  

    def forward(self, features_rgb, features_event):
        mean_rgb = torch.mean(features_rgb, dim=1)  
        B, D1, D2 = features_event.shape
        fe = features_event.repeat_interleave(6, dim=1) 
        fe = fe.view(B, 768, 128, 6).mean(dim=-1) 
        fe = fe.transpose(1, 2).contiguous() 
        processed_event, interacted_rgb = self.evrd_interact(fe, mean_rgb)  
        processed_event = processed_event.transpose(1, 2).contiguous()  
        processed_event = processed_event.view(B, 128, 6, 128).mean(dim=2)
        processed_event = processed_event.repeat_interleave(6, dim=2)
        interacted_rgb = interacted_rgb.unsqueeze(1).expand(-1, 128, -1)  
        features_rgb = features_rgb + interacted_rgb
        features_rgb = F.relu(features_rgb)

        features_event = features_event + processed_event 
        features_event = F.relu(features_event)

        x = features_rgb + features_event
        x, attn_weights = self.self_attn(x, x, x)  

        return x