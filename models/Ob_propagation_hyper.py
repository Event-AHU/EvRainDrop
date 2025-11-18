from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset
from torch.nn import init
import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import gather_csr, scatter, segment_csr


class Observation_progation(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int,
                 n_nodes: int, ob_dim: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0,** kwargs)  

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.n_nodes = n_nodes  

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.weight = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.bias = Parameter(torch.Tensor(heads * out_channels))

        self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, heads * out_channels))

        self.increase_dim = Linear(in_channels[1], heads * out_channels * 8)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))


        self.ob_dim = ob_dim
        self.hyperedge_index = None  

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
        glorot(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        glorot(self.nodewise_weights)
        glorot(self.map_weights)
        self.increase_dim.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], p_t: Tensor, hyperedge_index: Adj, 
                hyperedge_weights=None, use_beta=False, edge_attr: OptTensor = None, 
                return_attention_weights=None):
        r"""
        hyperedge
        """
        self.hyperedge_index = hyperedge_index  
        self.p_t = p_t
        self.use_beta = use_beta

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(hyperedge_index, x=x, edge_weights=hyperedge_weights, 
                            edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None
        hyperedge_index = self.hyperedge_index

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(hyperedge_index, Tensor):
                return out, (hyperedge_index, alpha)
            elif isinstance(hyperedge_index, SparseTensor):
                return out, hyperedge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        use_beta = self.use_beta
        if use_beta:
            n_step = self.p_t.shape[0]
            n_hyperedges = x_i.shape[0]
            h_W = self.increase_dim(x_i).view(-1, n_step, 32)
            w_v = self.map_weights[self.hyperedge_index[1]].unsqueeze(1)
            p_emb = self.p_t.unsqueeze(0)
            aa = torch.cat([w_v.repeat(1, n_step, 1), p_emb.repeat(n_hyperedges, 1, 1)], dim=-1)
            beta = torch.mean(h_W * aa, dim=-1)

        if edge_weights is not None:
            if use_beta:
                gamma = beta * edge_weights.unsqueeze(-1)
                gamma = torch.repeat_interleave(gamma, self.ob_dim, dim=-1)
                all_edge_weights = torch.mean(gamma, dim=1)
                K = int(gamma.shape[0] * 0.5)
                top_idx = torch.argsort(all_edge_weights, descending=True)[:K]
                gamma = gamma[top_idx]
                self.hyperedge_index = self.hyperedge_index[:, top_idx]
                index = self.hyperedge_index[0]
                x_i = x_i[top_idx]
                x_j = x_j[top_idx]
                if edge_attr is not None:
                    edge_attr = edge_attr[top_idx]
            else:
                gamma = edge_weights.unsqueeze(-1)
        else:
            gamma = torch.ones(x_i.size(0), 1, device=x_i.device)

        self.index = index
        self._alpha = torch.mean(gamma, dim=-1) if use_beta else gamma.squeeze(-1)
        
        gamma = softmax(gamma, index, ptr, num_nodes=self.n_nodes)  
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        out_obs = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.out_channels)
        
        gamma_scalar = gamma.mean(dim=1, keepdim=True)
        gamma_obs = gamma_scalar.view(-1, self.heads, 1)
        out_obs = out_obs * gamma_obs

        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        
        if self.lin_edge is not None and edge_attr is not None:
            edge_emb = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key = key + edge_emb
            
        alpha_sa = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        if edge_weights is not None and not use_beta:
            alpha_sa = edge_weights.unsqueeze(-1)
        alpha_sa = softmax(alpha_sa, index, ptr, num_nodes=self.n_nodes) 
        alpha_sa = F.dropout(alpha_sa, p=self.dropout, training=self.training)
        
        out_sa = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        out_sa = out_sa * alpha_sa.view(-1, self.heads, 1)

        out = out_obs + out_sa
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""超图聚合"""
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={}, n_nodes={})'.format(
            self.__class__.__name__, self.in_channels,
            self.out_channels, self.heads, self.n_nodes
        )
    