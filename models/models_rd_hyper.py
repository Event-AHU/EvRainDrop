import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset

from .transformer_conv import TransformerConv
from .Ob_propagation_hyper import Observation_progation
import warnings
import numbers

class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1) 
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe

class EvRaindrop_v1(nn.Module):
    def __init__(self, d_inp=768, d_model=1536, nhead=4, nhid=768, nlayers=2, dropout=0.3, max_len=215, d_static=768,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure=None, sensor_wise_mask=False, static=False,
                 d_two=False):  
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask

        self.d_inp = d_inp  
        self.d_static = d_static  
        self.total_nodes = d_inp + d_static  
        self.d_model = d_model
        self.static = static
        self.d_two = d_two 

        if self.static:
            self.emb = nn.Linear(d_static, d_inp)  
            self.static_interact = nn.Linear(d_static, d_static) 
        self.d_ob = int(d_model / d_inp)  

        self.encoder = nn.Linear(d_inp * self.d_ob, self.d_inp * self.d_ob)
        self.pos_encoder = PositionalEncodingTF(16, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_model + 16, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).cuda()

        self.ob_propagation = Observation_progation(
            in_channels=max_len * self.d_ob, 
            out_channels=max_len * self.d_ob, 
            heads=1,
            n_nodes=self.total_nodes,  
            ob_dim=self.d_ob
        ).cuda()

        self.ob_propagation_layer2 = Observation_progation(
            in_channels=max_len * self.d_ob, 
            out_channels=max_len * self.d_ob, 
            heads=1,
            n_nodes=self.total_nodes, 
            ob_dim=self.d_ob
        ).cuda()

        self.cluster_layer = nn.Linear(self.d_static * self.d_ob, self.d_inp).cuda()

        if self.d_two:
            shared_dim = self.d_ob  
            self.proj_dyn = nn.Linear(max_len * self.d_ob, shared_dim).cuda()
            self.proj_stat = nn.Linear(max_len * self.d_ob, shared_dim).cuda()
            self.Ap = None 

        if static == False:
            d_final = d_model + 16
        else:
            d_final = d_model + 16 + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

        self.cuda()
        self._ensure_all_params_on_gpu()

    def _ensure_all_params_on_gpu(self):
        for name, param in self.named_parameters():
            if param.device.type != 'cuda':
                param.data = param.data.cuda()
                if param.grad is not None:
                    param.grad.data = param.grad.data.cuda()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
            self.static_interact.weight.data.uniform_(-initrange, initrange)  

        glorot(self.cluster_layer.weight)
        zeros(self.cluster_layer.bias)
        if self.d_two:
            glorot(self.proj_dyn.weight)
            zeros(self.proj_dyn.bias)
            glorot(self.proj_stat.weight)
            zeros(self.proj_stat.bias)

    def build_hypergraph(self, static_features, dynamic_features=None):
        batch_size = static_features.shape[0]
        if self.d_two and (dynamic_features is not None):
            hyperedge_list_per_batch = []
            Ap_list = []  
            for b in range(batch_size):
                stat_b = static_features[b]  
                dyn_b  = dynamic_features[b]   

                stat_emb = self.proj_stat(stat_b)   
                dyn_emb  = self.proj_dyn(dyn_b)     

                affinity = torch.matmul(stat_emb, dyn_emb.transpose(0, 1)) 
                probs = F.softmax(affinity, dim=0)  
                Ap_list.append(probs.detach())

                batch_hyperedges = []
                for core_node in range(self.d_inp):  
                    top_static_nodes = torch.topk(probs[:, core_node], k=6, dim=0).indices  
                    core_idx = torch.full((6,), core_node, device=static_features.device)  
                    static_idx = top_static_nodes + self.d_inp  
                    assert static_idx.max() < self.total_nodes, \
                        f"静态节点索引越界: 最大值 {static_idx.max()} 超过总节点数 {self.total_nodes}"
                    hyperedge = torch.stack([core_idx, static_idx], dim=0) 
                    batch_hyperedges.append(hyperedge)

                batch_hyperedge_index = torch.cat(batch_hyperedges, dim=1)  
                hyperedge_list_per_batch.append(batch_hyperedge_index)

            self.Ap = torch.stack(Ap_list, dim=1) 
            return hyperedge_list_per_batch  

        static_nodes = static_features.permute(1, 0, 2) 
        cluster_scores = self.cluster_layer(static_nodes)
        cluster_probs = F.softmax(cluster_scores, dim=-1) 
        

        hyperedge_list_per_batch = []
        for b in range(batch_size):  
            batch_hyperedges = []
            for core_node in range(self.d_inp):  
                static_probs = cluster_probs[:, b, core_node] 
                top_static_nodes = torch.topk(static_probs, k=6, dim=0).indices  
                
                core_idx = torch.full((6,), core_node, device=static_features.device)  

                static_idx = top_static_nodes + self.d_inp  
                
                assert static_idx.max() < self.total_nodes, \
                    f"静态节点索引越界: 最大值 {static_idx.max()} 超过总节点数 {self.total_nodes}"
                
                hyperedge = torch.stack([core_idx, static_idx], dim=0)
                batch_hyperedges.append(hyperedge)
            
            batch_hyperedge_index = torch.cat(batch_hyperedges, dim=1)
            hyperedge_list_per_batch.append(batch_hyperedge_index)
        
        return hyperedge_list_per_batch 

    def forward(self, src, static, times, lengths):
        maxlen, batch_size = src.shape[0], src.shape[1]
        missing_mask = torch.zeros_like(src[:, :, :1]) 
        n_sensor = self.d_inp

        target_dim = self.d_inp * self.d_ob
        current_dim = src.shape[-1]
        repeat_times = target_dim // current_dim
        src = torch.repeat_interleave(src, repeat_times, dim=-1)
        h = F.relu(src)  
        pe = self.pos_encoder(times)  

        static_node_features = None
        hyperedge_list = None  
        interacted_static = static.clone()  
        if static is not None and self.static:
            emb = self.emb(static)  
            static_node_features = static.unsqueeze(2)  
            static_node_features = static_node_features.repeat(1, 1, maxlen * self.d_ob)  
            static_node_features = self.relu(self.static_interact(static_node_features.permute(0,2,1)).permute(0,2,1)) 
            if self.d_two:
                dynamic_node_features = h.permute(1, 0, 2).reshape(batch_size, self.d_inp, -1)
                hyperedge_list = self.build_hypergraph(static_node_features, dynamic_node_features) 
            else:
                hyperedge_list = self.build_hypergraph(static_node_features)

        h = self.dropout(h)

        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(src.device)
        adj = self.global_structure.to(src.device) if self.global_structure is not None else torch.ones([self.d_inp, self.d_inp]).cuda()
        adj[torch.eye(self.d_inp).bool()] = 1 
        edge_index = torch.nonzero(adj).T 
        edge_weights = adj[edge_index[0], edge_index[1]]

        x = h
        n_step = src.shape[0]
        output = torch.zeros([n_step, batch_size, self.d_inp * self.d_ob]).cuda()

        for unit in range(batch_size):
            stepdata = x[:, unit, :].reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
            stepdata = stepdata.reshape(self.d_inp, n_step * self.d_ob)  
            
            stepdata, attentionweights = self.ob_propagation(
                stepdata, 
                p_t=pe[:, unit, :], 
                hyperedge_index=edge_index,
                hyperedge_weights=edge_weights,
                use_beta=False,
                return_attention_weights=True
            )
            
            stepdata = stepdata.view([self.d_inp, n_step, self.d_ob]).permute([1, 0, 2])
            stepdata = stepdata.reshape([-1, self.d_inp * self.d_ob])
            output[:, unit, :] = stepdata

        if static is not None and self.static and hyperedge_list is not None:
            core_node_features = output.permute(1, 0, 2).reshape(batch_size, self.d_inp, -1) 
            all_node_features = torch.cat([
                core_node_features,  
                static_node_features 
            ], dim=1)  
            
            updated_node_features = []
            for unit in range(batch_size):
                batch_hyperedge = hyperedge_list[unit]
                node_feats = all_node_features[unit]
                updated_feats = self.ob_propagation(
                    node_feats,
                    p_t=pe[:, unit, :],
                    hyperedge_index=batch_hyperedge,  
                    use_beta=False
                )
                updated_core_feats = updated_feats[:self.d_inp]  
                updated_static_feats = updated_feats[self.d_inp:] 
                interacted_static[unit] = updated_static_feats.mean(dim=1) 
                updated_node_features.append(updated_core_feats)
            
            stacked_feats = torch.stack(updated_node_features, dim=0) 
            feat_with_time = stacked_feats.view(batch_size, self.d_inp, n_step, self.d_ob)
            output = feat_with_time.permute(2, 0, 1, 3).reshape(n_step, batch_size, self.d_inp * self.d_ob)

        if static is not None and self.static and hyperedge_list is not None:
            static_node_features_2 = interacted_static.unsqueeze(2).repeat(1, 1, maxlen * self.d_ob)  
        
        for unit in range(batch_size):
            if static is not None and self.static and hyperedge_list is not None:
                core_feat = output[:, unit, :].reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2).reshape(self.d_inp, -1)
                static_feat = static_node_features_2[unit].reshape(self.d_static, -1)
                stepdata = torch.cat([core_feat, static_feat], dim=0)  
                batch_hyperedge = hyperedge_list[unit]
                hyperedge_weights = torch.ones(batch_hyperedge.shape[1], device=stepdata.device)
            else:
                stepdata = output[:, unit, :].reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2).reshape(self.d_inp, n_step * n_step * self.d_ob)

                interacted_static[unit] = updated_static_feats_2.mean(dim=1)  
            
            stepdata, attentionweights = self.ob_propagation_layer2(
                stepdata, 
                p_t=pe[:, unit, :], 
                hyperedge_index=batch_hyperedge, 
                hyperedge_weights=hyperedge_weights,  
                use_beta=False,
                return_attention_weights=True
            )

            stepdata = stepdata[:self.d_inp] 
            stepdata = stepdata.view([self.d_inp, n_step, self.d_ob]).permute([1, 0, 2])
            stepdata = stepdata.reshape([-1, self.d_inp * self.d_ob])
            output[:, unit, :] = stepdata

        if self.sensor_wise_mask:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1).view(-1, batch_size, self.d_inp*(self.d_ob+16))
        else:
            output = torch.cat([output, pe], axis=2)  

        r_out = self.transformer_encoder(output, src_key_padding_mask=mask)

        masked_agg = True
        if masked_agg:
            lengths2 = lengths.unsqueeze(1)
            mask2 = mask.permute(1, 0).unsqueeze(2).float()
            if self.sensor_wise_mask:
                output = torch.zeros([batch_size, self.d_inp, self.d_ob+16]).cuda()
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                for se in range(self.d_inp):
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16))
                    out = r_out[:, :, se, :]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        else:
            output = r_out[-1, :, :].squeeze(0)

        if static is not None and self.static:
            output = torch.cat([output, emb], dim=1)
        output = self.mlp_static(output)

        return output, interacted_static, None