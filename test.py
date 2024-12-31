from utils.dataset import DatasetFormatter
from torch_geometric.loader import DataLoader

dataset = DatasetFormatter(max_epochs=5, dataset_dir="dataset_split")
train_loader = DataLoader(dataset=dataset, batch_size=8, num_workers=4)

for batch_index, graph in enumerate(train_loader):

        # graph = transformer(graph)
        print(f"loop dataloader: {batch_index}")
        graph = graph.cuda()
        print(graph.x)
        # node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
        # velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        # predicted_acc, target_acc = model(graph, velocity_sequence_noise)
        # mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)
        
        # errors = ((predicted_acc - target_acc)**2)[mask]
        # loss = torch.mean(errors)