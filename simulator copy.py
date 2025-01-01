from model import GMN
import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils.norm import Normalizer
import os

class Simulator(nn.Module):

    def __init__(self, 
                 message_passing_num, 
                 node_input_size, 
                 edge_input_size, 
                 latent_dim_size = 16, 
                 device = 'cuda:0', 
                 model_dir='checkpoint/simulator.pth', 
                 training = True) -> None:
        
        super(Simulator, self).__init__()
        self.device = device
        self.training = training
        self.node_input_size =  node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.model = GMN(message_passing_num=message_passing_num, node_attr_size=node_input_size, edge_attr_size=edge_input_size, latent_dim_size=latent_dim_size).to(device)
        self._output_normalizer = Normalizer(size=2, name='output_normalizer', device=device)
        self._node_normalizer = Normalizer(size=2, name='node_normalizer', device=device) # sizenya 2 velocity + 4 possible cell type
        # self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        print('Simulator model initialized')

    def update_node_attr(self, frames, types:torch.Tensor):
        node_feature = []

        node_feature.append(frames) #velocity
        node_feature = torch.cat(node_feature, dim=1).to(self.device)
        node_feature = self._node_normalizer(node_feature, self.training)

        cell_type = torch.squeeze(types.long()).to(self.device)
        one_hot = torch.nn.functional.one_hot(cell_type, 4).float()
        one_hot = one_hot[:, 1:2].to(self.device) # only taking the solid
        node_feature = torch.cat((node_feature, one_hot), dim = 1).to(self.device)

        return node_feature

    def velocity_to_accelation(self, noised_frames, next_velocity):

        acc_next = next_velocity - noised_frames
        return acc_next


    def forward(self, graph:Data, velocity_sequence_noise = None):

        if self.training:
            

            cell_type = graph.x[:, 0:1] # cell types
            frames = graph.x[:, 1:3] # x and y velocoity
            target = graph.y
            
            if velocity_sequence_noise is not None:
                frames = frames + velocity_sequence_noise
            
            node_attr = self.update_node_attr(frames, cell_type)

            graph.x = node_attr
            predicted = self.model(graph)

            target_acceration = self.velocity_to_accelation(frames, target)
            target_acceration_normalized = self._output_normalizer(target_acceration, self.training)

            return predicted, target_acceration_normalized

        else:
            
            cell_type = graph.x[:, 0:1] # cell type
            frames = graph.x[:, 1:3] # x and y velocoity
            node_attr = self.update_node_attr(frames, cell_type)
            graph.x = node_attr

            predicted = self.model(graph)

            velocity_update = self._output_normalizer.inverse(predicted)
            predicted_velocity = frames + velocity_update

            return predicted_velocity

    def load_checkpoint(self, ckpdir=None):
        
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir)
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.'+k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s"%ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir=self.model_dir

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer  = self._node_normalizer.get_variable()
        # _edge_normalizer = self._edge_normalizer.get_variable()

        to_save = {'model':model, '_output_normalizer':_output_normalizer, '_node_normalizer':_node_normalizer}

        torch.save(to_save, savedir)
        print('Simulator model saved at %s'%savedir)