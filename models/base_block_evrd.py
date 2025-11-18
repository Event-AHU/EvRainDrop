import torchvision.models as torch_models
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *
from models.vrwkv import *
from clip import clip
from mmcls.models import build_backbone, build_classifier
from mmcls_custom.models.backbones.GCN import * 
from config import argument_parser

from models.evraindrop import EvRainDrop

parser = argument_parser()
args = parser.parse_args()
class TransformerClassifier(nn.Module):
    def __init__(self, attr_num, dim=768, pretrain_path='/wdata/Code/ZF/evrd_par/pretrain/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        self.attr_num = attr_num
        self.dim = dim
        self.mode=args.fusion_mode
        self.backbones=args.backbones

        if self.backbones=='rwkv':
            self.vit = vit_base()
            self.vit.load_param(pretrain_path)

            self.vrwkv_rgb = VRWKV(args.rwkv_config,args.rwkv_checkpoint)
            self.vrwkv_event = VRWKV(args.rwkv_config,args.rwkv_checkpoint)
            self.blocks =self.vit.blocks[-1:]
        elif self.backbones=='vit':
            self.vit_rgb = vit_base()
            self.vit_event = vit_base()
            self.vit_rgb .load_param(pretrain_path)
            self.vit_event.load_param(pretrain_path)
            self.blocks =self.vit_rgb.blocks[-1:]
            
        elif self.backbones=='resnet50':
            self.vit = vit_base()
            self.vit.load_param(pretrain_path)
            self.resnet50 = torch_models.resnet50(pretrained=True)
            self.resnet50 = self.resnet50.eval() 
            self.resnet50_feature_extractor = nn.Sequential(*list(self.resnet50.children())[:-3])
            self.rgb_proj = nn.Linear(1024, 768)
            self.event_proj = nn.Linear(1024, 768)
            self.blocks =self.vit.blocks[-1:]


        self.norm = nn.LayerNorm(dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)
       
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=128)

        self.gcn = TokenGCN(num_layers=3, use_knn=True)

        self.evraindrop = EvRainDrop()
        # self.evraindrop = EVRAindrop(
        #    evrd_kwargs={
        #     "nhead": 2, 
        #     "nlayers": 1,
        #     }
        # )
       
    def forward(self, imgs, word_vec):

        if self.backbones=='rwkv': 
            features_rgb = self.vrwkv_rgb.forward(imgs[0]).to("cuda").float()
            features_rgb =features_rgb.mean(1)
            features_event_5 = self.vrwkv_event.forward(imgs[1]).to("cuda").float()
            features_rgb = features_rgb.squeeze(1)

            B,T,N,D=features_event_5.shape
            features_event_5=features_event_5.reshape(B, T * N, D)
            features_event=self.filter_similar_t_vectorized_2(features_event_5)
            assert features_event.shape[1]>=128, f'filter features_event less 128'

            if features_event.shape[1] > 128:
                features_event=features_event.transpose(1, 2)
                features_event=self.adaptive_pool(features_event)
                features_event=features_event.transpose(1, 2)
        x = self.evraindrop(features_rgb, features_event)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)

        return logits
    

    def filter_similar_t_vectorized_2(self, x, threshold=0.75, min_t=128):
        B, TN, D = x.shape

        x_norm = x / torch.norm(x, dim=-1, keepdim=True)  
        sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2))  

        mask = torch.ones((B, TN), dtype=torch.bool, device=x.device)

        high_sim = sim_matrix > threshold
        high_sim &= ~torch.eye(TN, dtype=torch.bool, device=x.device).unsqueeze(0)

        mask &= ~high_sim.any(dim=2)  

        for i in range(B):
            if mask[i].sum() < min_t:
                scores = torch.norm(x[i], dim=-1)  
                topk_indices = scores.topk(min_t, largest=True).indices

                mask[i] = False  
                mask[i][topk_indices] = True 

        filtered_t = [x[i][mask[i]] for i in range(B)]
        filtered_t = torch.nn.utils.rnn.pad_sequence(filtered_t, batch_first=True)

        return filtered_t

    
