import torch
import torch.nn as nn
from torch_geometric.data import Data
from utils.utils import decompose_graph, copy_geometric_data
from torch_scatter import scatter_add

def build_MLP(input_dim: int, hidden_layers: int, output_dim: int, normalization: bool = True) -> nn.Sequential:
    module = nn.Sequential(nn.Linear(input_dim, hidden_layers), 
                           nn.ReLU(), 
                           nn.Linear(hidden_layers, hidden_layers), 
                           nn.ReLU(), 
                           nn.Linear(hidden_layers, hidden_layers), 
                           nn.ReLU(), 
                           nn.Linear(hidden_layers, output_dim))
    if normalization:
        return nn.Sequential(module,  nn.LayerNorm(normalized_shape=output_dim))
    return module

#Encoder Class
class Encoder(nn.Module):
    def __init__(self, node_attr_size: int, latent_dim_size: int) -> None:
        super(Encoder, self).__init__()
        self.node_encoder = build_MLP(node_attr_size, latent_dim_size, latent_dim_size)

    def forward(self, graph: Data) -> Data:
        node_attr, edge_index, _, _ = decompose_graph(graph)  # Ignore edge_attr
        node_ = self.node_encoder(node_attr)
        return Data(x=node_, edge_index=edge_index)
    
#Message Passing Processor Class
class MessagePassing(nn.Module):
    def __init__(self, latent_dim_size: int) -> None:
        super(MessagePassing, self).__init__()

        # Update dimensions since edge attributes are removed
        node_input_dim = 2 * latent_dim_size

        self.node_updater = build_MLP(input_dim=node_input_dim, hidden_layers=latent_dim_size, output_dim=latent_dim_size)

    def forward(self, graph: Data) -> Data:
        node_attr, edge_index, _, _ = decompose_graph(graph)  # Ignore edge_attr
        senders_index, receivers_index = edge_index

        # Node message passing
        num_nodes = graph.num_nodes
        agg_received_edges = scatter_add(node_attr[senders_index], receivers_index, dim=0, dim_size=num_nodes)

        # Update node attributes
        nodes_to_collect = [graph.x, agg_received_edges]
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        x_updated = self.node_updater(collected_nodes)
        return Data(x=x_updated, edge_index=edge_index)
    
#Decoder Class
class Decoder(nn.Module):
    def __init__(self, latent_dim_size: int, output_size: int) -> None:
        super(Decoder, self).__init__()
        #Initialize decoder
        self.decoder = build_MLP(latent_dim_size, latent_dim_size, output_size, normalization=False)

    def forward(self, graph: Data) -> Data:
        return self.decoder(graph.x)
    
#GMN Class
class GMN(nn.Module):
    def __init__(self, node_attr_size: int, latent_dim_size: int, message_passing_num: int, normalization: bool = True) -> None:
        super(GMN, self).__init__()

        # Encoder
        self.encoder = Encoder(node_attr_size=node_attr_size, latent_dim_size=latent_dim_size)

        # Message passing processors with multiple message passing
        processer_list = [MessagePassing(latent_dim_size=latent_dim_size) for _ in range(message_passing_num)]
        self.processer_list = nn.ModuleList(processer_list)

        # Decoder
        self.decoder = Decoder(latent_dim_size=latent_dim_size, output_size=node_attr_size)

    def forward(self, graph: Data) -> Data:
        encoded_graph = self.encoder(graph)
        for message_passing in self.processer_list:
            encoded_graph = message_passing(encoded_graph)
        decoded_graph = self.decoder(encoded_graph)
        return decoded_graph[:, 1:3]  # Only returning the velocity
