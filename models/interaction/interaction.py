import torch
from torch.nn import Module, ModuleList, LeakyReLU, LayerNorm
from torch_scatter import scatter_sum
from math import pi as PI

from ..model_utils import GaussianSmearing, EdgeExpansion
from ..invariant import GVLinear, VNLeakyReLU, MessageModule


class InteractionModule(Module):
    def __init__(self, node_sca_dim=256, node_vec_dim=64, edge_dim=64,hid_dim=128,num_geodesic=2, \
        num_geoattn=4, k=24, cutoff=10.):

        super().__init__()

        self.node_sca_dim = node_sca_dim
        self.node_vec_dim = node_vec_dim
        self.edge_dim = edge_dim 
        self.hid_dim = hid_dim 
        self.num_geodesic = num_geodesic
        self.num_geoattn = num_geoattn
        self.k = k
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_geodesic):
            block = Geodesic_GNN(
                node_sca_dim=node_sca_dim,
                node_vec_dim=node_vec_dim,
                hid_dim = hid_dim,
                edge_dim = edge_dim,
                num_edge_types=2, 
                out_sca_dim=node_sca_dim,
                out_vec_dim=node_vec_dim,
                cutoff=cutoff
            )
            self.interactions.append(block)

        for _ in range(num_geoattn):
            block = Geoattn_GNN(
                node_sca_dim=node_sca_dim,
                node_vec_dim=node_vec_dim,
                hid_dim = hid_dim,
                edge_dim = edge_dim,
                num_edge_types=4, 
                out_sca_dim=node_sca_dim,
                out_vec_dim=node_vec_dim,
                cutoff=cutoff
            )
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]
    
    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, idx_ligand, idx_surface, gds_edge_index, gds_edge_feature, gds_dis, geom_edge_index, geom_edge_feature):
        
        h_surface_sca = node_attr[0][idx_surface]
        h_surface_vec = node_attr[1][idx_surface]
        gds_edge_vec = pos[idx_protein][gds_knn_edge_index[0]]-pos[idx_protein][gds_knn_edge_index[1]]

        for geodesic_block in self.interactions[:self.num_geodesic]:
            delta_h = geodesic_block([h_surface_sca,h_surface_vec], gds_edge_feature, gds_edge_vec, gds_edge_index, gds_dis)
            h_surface_sca = h_surface_sca + delta_h[0]
            h_surface_vec = h_surface_vec + delta_h[1]

        h_ligpkt_sca = torch.cat([node_attr[0][data.idx_ligand], h_surface_sca], dim=0)
        h_ligpkt_vec = torch.cat([node_attr[1][data.idx_ligand], h_surface_vec], dim=0)
        geom_edge_vec = pos[geom_edge_index[0]] - pos[geom_edge_index[1]]

        for geoattn_block in self.interactions[self.num_geoattn:]:
            delta_h = geoattn_block([h_ligpkt_sca,h_ligpkt_vec], geom_edge_feature, geom_edge_vec, geom_edge_index)
            h_ligpkt_sca = h_ligpkt_sca + delta_h[0]
            h_ligpkt_vec = h_ligpkt_vec + delta_h[1]

        return [h_ligpkt_sca, h_ligpkt_vec]



##############################################################################################################
class CFTransformerEncoderVN(Module):
    
    def __init__(self, hidden_channels=[256, 64], edge_channels=64, num_edge_types=4, key_channels=128, num_heads=4, num_interactions=6, k=32, cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels  # not use
        self.num_heads = num_heads  # not use
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlockVN(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                num_edge_types=num_edge_types,
                key_channels=key_channels,
                num_heads=num_heads,
                cutoff = cutoff
            )
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]
    
    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, edge_index, edge_feature):

        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]

        h = list(node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, edge_index, edge_feature, edge_vector)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return h


class AttentionInteractionBlockVN(Module):

    def __init__(self, hidden_channels, edge_channels, num_edge_types, key_channels, num_heads=1, cutoff=10.):
        super().__init__()
        self.num_heads = num_heads
        # edge features
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels - num_edge_types)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)
        ## compare encoder and classifier message passing

        # edge weigths and linear for values
        self.message_module = MessageModule(hidden_channels[0], hidden_channels[1], edge_channels, edge_channels,
                                                                                hidden_channels[0], hidden_channels[1], cutoff)

        # centroid nodes and finall linear
        self.centroid_lin = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(hidden_channels[1])
        self.out_transform = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])

        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    def forward(self, x, edge_index, edge_feature, edge_vector):
        """
        Args:
            x:  Node features: scalar features (N, feat), vector features(N, feat, 3)
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        scalar, vector = x
        N = scalar.size(0)
        row, col = edge_index   # (E,) , (E,)

        # Compute edge features
        edge_dist = torch.norm(edge_vector, dim=-1, p=2)
        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1)
        edge_vec_feat = self.vector_expansion(edge_vector) 

        msg_j_sca, msg_j_vec = self.message_module(x, (edge_sca_feat, edge_vec_feat), col, edge_dist, annealing=True)

        # Aggregate messages
        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=N)  #.view(N, -1) # (N, heads*H_per_head)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=N)  #.view(N, -1, 3) # (N, heads*H_per_head, 3)
        x_out_sca, x_out_vec = self.centroid_lin(x)
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out

