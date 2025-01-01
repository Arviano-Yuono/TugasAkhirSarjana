from torch.utils.data import IterableDataset
import os, numpy as np
import os.path as osp
import h5py
from torch_geometric.data import Data
import torch
import math
import time
from typing import Union
from torch_geometric.utils import grid

class DatasetFormat():
    def __init__(self, max_epochs: int = 50, 
                 files: Union[h5py.File, None] = None, 
                 open_trajectory: int = 5,
                 device="cuda:0",
                 x_size: int = 802,
                 y_size: int = 402) -> None:

        # time_interval: float = 0.01,
        self.device = device
        self.open_tra_num = open_trajectory
        self.file_handle = files
        self.tra_ori_list = list(self.file_handle.keys())
        self.processed_tra = set()  # Track processed trajectories
        self.shuffle_file()
        # exclued last trajectory
        if self.tra_ori_list:
            # print(f"{self.tra_ori_list}\nujungnya: {len(self.tra_ori_list)-1}")
            self.processed_tra.add(str(len(self.tra_ori_list)-1))

        self.data_keys = ("cell_type", "x_velocity", "y_velocity")
        # self.out_keys = list(self.data_keys) + ['time']
        self.tra_index = 0
        self.epcho_num = 1
        self.tra_readed_index = -1

        self.tra_len = len(self.file_handle)
        # self.time_iterval = time_interval

        self.opened_tra = []
        self.opened_tra_readed_index = {}
        self.tra_data = {}
        self.max_epochs = max_epochs

        #constant attributes
        self.edge_index, self.pos = grid(height=y_size, width=x_size, device=self.device)
        self.edge_attr = torch.ones(self.edge_index.size(1), 1).to(self.device)  # Shape [num_edges, 1]
        self.cell_type_attr = None

    def open_tra(self):
        """
        Open new trajectories and ensure no repetition within an epoch.
        """
        while len(self.opened_tra) < self.open_tra_num:
            # Check if all trajectories have been processed
            if self.tra_index >= len(self.datasets)-1:
                self.epcho_end()
                print('Epoch Finished')
                return

            # Skip processed trajectories and the last trajectory
            if self.datasets[self.tra_index] in self.processed_tra:
                self.tra_index += 1
                continue

            # Add the trajectory if valid
            tra_index = self.datasets[self.tra_index]
            self.opened_tra.append(tra_index)
            self.opened_tra_readed_index[tra_index] = -1
            self.tra_index += 1


    def check_and_close_tra(self):
        """
        Close trajectories that are fully processed.
        """
        to_del = []
        for tra in self.opened_tra:
            if self.opened_tra_readed_index[tra] >= (self.tra_len - 3):
                to_del.append(tra)
                self.processed_tra.add(str(len(self.tra_ori_list)-1))  # Mark trajectory as processed

        for tra in to_del:
            self.opened_tra.remove(tra)
            del self.opened_tra_readed_index[tra]
            if tra in self.tra_data:
                del self.tra_data[tra]

    def shuffle_file(self):
        """
        Shuffle the dataset files for randomness.
        """
        self.processed_tra.clear()  # Clear processed trajectories for the new epoch
        self.tra_index = 0
        datasets = list(self.file_handle.keys())
        np.random.shuffle(datasets)
        self.datasets = datasets

    def epcho_end(self):
        """
        Handle end of epoch logic.
        """
        self.tra_index = 0
        self.opened_tra = []
        self.opened_tra_readed_index = {}
        self.tra_data = {}
        self.processed_tra.clear()
        self.processed_tra.add(str(len(self.tra_ori_list)-1))

        self.shuffle_file()
        self.epcho_num += 1

    # @staticmethod
    def datas_to_graph(self, datas, device = "cuda:0") -> Data:
        
        x_velocity_attr = torch.from_numpy(np.array(datas[1][0])).float().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]
        y_velocity_attr = torch.from_numpy(np.array(datas[2][0])).float().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]
        x_velocity_attr_target = torch.from_numpy(np.array(datas[1][1])).float().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]
        y_velocity_attr_target = torch.from_numpy(np.array(datas[2][1])).float().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]
        # time_vector = torch.ones((len(datas[0]), 1), dtype=torch.float32, device=device) * torch.tensor(datas[4], device=device)  # Shape: [num_nodes, 1]

        #check if cell type exist
        if self.cell_type_attr is None:
            self.cell_type_attr = torch.from_numpy(np.array(datas[0])).int().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]

        node_attr = torch.cat((self.cell_type_attr, 
                               x_velocity_attr, 
                               y_velocity_attr), 
                               dim=1).to(device) # Shape: [num_nosed, 4]

        target = torch.cat((
            x_velocity_attr_target,  # x_velocity target
            y_velocity_attr_target   # y_velocity target
        ), dim=1).to(device) # Shape: [num_nosed, 2]
        
        g = Data(x=node_attr, y=target, pos=self.pos, edge_index=self.edge_index, edge_attr=self.edge_attr)
        torch.cuda.empty_cache()
        
        return g
    
    def __next__(self) -> Data:
        """
        Return the next trajectory data.
        """
        print('next')
        self.check_and_close_tra()
        self.open_tra()

        if self.epcho_num > self.max_epochs:
            raise StopIteration

        if not self.opened_tra:
            if self.tra_index >= len(self.datasets):
                self.epcho_end()  # Reset for the next epoch
                raise StopIteration(f"All trajectories processed. Moving to epoch {self.epcho_num}.")
            raise StopIteration("No trajectories available to process.")

        # Select the next trajectory in sequence
        selected_tra = self.opened_tra.pop(0)
        self.opened_tra_readed_index[selected_tra] += 1

        data = self.tra_data.get(selected_tra)
        if data is None:
            data = self.file_handle[selected_tra]
            self.tra_data[selected_tra] = data

        # print(f"selected trajectory: {selected_tra}")

        # Get the target trajectory
        target_tra = str(int(selected_tra) + 1)
        target = self.tra_data.get(target_tra)
        if target is None:
            target = self.file_handle[target_tra]

        # print(f"target trajectory: {target_tra}")

        datas = []
        for k in self.data_keys:
            if k in ["x_velocity", "y_velocity"]:
                r = np.array((data[k], target[k]), dtype=np.float32)
            else:
                r = data[k]
                if k == "cell_type":
                    r = r.astype(np.int32)
            datas.append(r)
        # datas.append(np.array([self.time_iterval], dtype=np.float32))

        # print(f"Allocated before: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        # print(f"Cached before: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        g = self.datas_to_graph(datas, self.device)
        # print(f"Allocated after: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        # print(f"Cached after: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        return g

class DatasetFormatter(IterableDataset):
    def __init__(self, max_epochs: int, 
                 dataset_dir: str, 
                 split: str='train',
                 open_tra_num: int = 5,
                 device = "cuda:0") -> None:
        super().__init__()

        self.device = device
        self.max_epochs= max_epochs
        self.open_tra_num = open_tra_num
        self.dataset_dir = osp.join(dataset_dir, split+'.h5')

        #Check if dataset exist
        assert os.path.isfile(self.dataset_dir), '%s not exist' % dataset_dir

        print('Dataset '+  self.dataset_dir + ' Initilized')

    def __iter__(self) -> DatasetFormat:

        self.file_handle = h5py.File(self.dataset_dir, "r", swmr=True)
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_handle)
        else:
            per_worker = int(math.ceil(len(self.file_handle)/float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_handle))

        keys = list(self.file_handle.keys())
        keys = keys[iter_start:iter_end]
        files = {k: self.file_handle[k] for k in keys}
        return DatasetFormat(max_epochs=self.max_epochs, 
        open_trajectory=self.open_tra_num,
        files=files, 
        device=self.device)

# class DatasetFormat():
#     def __init__(self, max_epochs: int=50, 
#                  files: Union[h5py.File, None] = None, 
#                  time_interval:float = 0.01,
#                  open_trajectory: int = 5,
#                  device = "cuda:0") -> None:
        
#         self.device = device
#         self.open_tra_num = open_trajectory
#         self.file_handle=files
#         self.tra_ori_list =list(self.file_handle.keys())
#         self.shuffle_file()

#         self.data_keys =  ("cell_type", "density", "pos", "x_velocity", "y_velocity")
#         self.out_keys = list(self.data_keys)  + ['time']
#         self.tra_index = 0
#         self.epcho_num = 1
#         self.tra_readed_index = -1
#         # dataset attr
#         # treajectory_len: the time step between trajectory
#         self.tra_len = self.file_handle.__len__()
#         self.time_iterval = time_interval
        
#         self.opened_tra = []
#         self.opened_tra_readed_index = {}
#         self.opened_tra_readed_random_index = {}
#         self.tra_data = {}
#         self.max_epochs = max_epochs
        
    
#     def open_tra(self):
#         while(len(self.opened_tra) < self.open_tra_num):

#             tra_index = self.datasets[self.tra_index]

#             if tra_index not in self.opened_tra:
#                 self.opened_tra.append(tra_index)
#                 self.opened_tra_readed_index[tra_index] = -1
#                 self.opened_tra_readed_random_index[tra_index] = np.random.permutation(self.tra_len - 2)

#             self.tra_index += 1

#             if self.check_if_epcho_end():
#                 self.epcho_end()
#                 print('Epoch Finished')
    
#     def check_and_close_tra(self) -> None:
#         to_del = []

#         for tra in self.opened_tra:
#             if self.opened_tra_readed_index[tra] >= (self.tra_len - 3):
#                 to_del.append(tra)

#         for tra in to_del:
#             self.opened_tra.remove(tra)
#             try:
#                 del self.opened_tra_readed_index[tra]
#                 del self.opened_tra_readed_random_index[tra]
#                 del self.tra_data[tra]
#             except Exception as e:
#                 print(e)

#     def shuffle_file(self) -> None:
#         datasets = list(self.file_handle.keys())
#         np.random.shuffle(datasets)
#         self.datasets = datasets

#     def epcho_end(self) -> None:
#         self.tra_index = 0
#         self.shuffle_file()
#         self.epcho_num = self.epcho_num + 1

#     def check_if_epcho_end(self) -> bool:
#         if self.tra_index >= len(self.file_handle):
#             return True
#         return False

#     @staticmethod
#     def datas_to_graph(datas, device = "cuda:0") -> Data:

#         cell_type_attr = torch.from_numpy(np.array(datas[0])).int().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]
#         x_velocity_attr = torch.from_numpy(np.array(datas[3][0])).float().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]
#         y_velocity_attr = torch.from_numpy(np.array(datas[4][0])).float().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]
#         x_velocity_attr_target = torch.from_numpy(np.array(datas[3][1])).float().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]
#         y_velocity_attr_target = torch.from_numpy(np.array(datas[4][1])).float().unsqueeze(-1).to(device)  # Shape: [num_nodes, 1]
#         time_vector = torch.ones((len(datas[0]), 1), dtype=torch.float32, device=device) * torch.tensor(datas[5], device=device)  # Shape: [num_nodes, 1]

#         node_attr = torch.cat((cell_type_attr, 
#                                x_velocity_attr, 
#                                y_velocity_attr, 
#                                time_vector), 
#                                dim=1).to(device) # Shape: [num_nosed, 4]

#         target = torch.cat((
#             x_velocity_attr_target,  # x_velocity target
#             y_velocity_attr_target   # y_velocity target
#         ), dim=1).to(device) # Shape: [num_nosed, 2]

#         edge_index, pos = grid(height=402, width=802, device=device)
#         edge_attr = torch.ones(edge_index.size(1), 1).to(device)  # Shape [num_edges, 1]
        
#         g = Data(x=node_attr, y=target, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
#         torch.cuda.empty_cache()
#         return g
    
#     def __next__(self) -> Data:
#         self.check_and_close_tra()
#         self.open_tra()
        
#         if self.epcho_num > self.max_epochs:
#             raise StopIteration

#         # avoid selecting the last trajectory
#         valid_tra = [tra for tra in self.opened_tra if self.tra_ori_list.index(tra) < len(self.tra_ori_list) - 1]
#         if not valid_tra:
#             raise StopIteration

#         selected_tra = np.random.choice(valid_tra)

#         data = self.tra_data.get(selected_tra, None)

#         if data is None:
#             data = self.file_handle[selected_tra]
#             self.tra_data[selected_tra] = data
#         print(f"selected tarjectory: {selected_tra}")

#         # Get the next trajectory as the target
#         target_tra = str(int(selected_tra) + 1)
#         target = self.tra_data.get(target_tra, None)
#         print(f"target tarjectory: {target_tra}")

#         if target is None:
#             target = self.file_handle[target_tra]

#         # selected_tra_readed_index = self.opened_tra_readed_index[selected_tra]
#         self.opened_tra_readed_index[selected_tra] += 1

#         datas = []

#         for k in self.data_keys:
#             if k in ["density", "x_velocity", "y_velocity"]:
#                 r = np.array((data[k], target[k]), dtype=np.float32)
#             else:
#                 r = data[k]
#                 if k in ["cell_type"]:
#                     r = r.astype(np.int32)
#             datas.append(r)
#         datas.append(np.array([self.time_iterval], dtype=np.float32))

#         g = self.datas_to_graph(datas, self.device)
#         return g