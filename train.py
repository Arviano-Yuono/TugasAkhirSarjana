from utils.dataset import DatasetFormatter
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
from utils.noise import get_velocity_noise
from simulator import Simulator
import torch
from utils.utils import NodeType

noise_std=2e-2
# Define the model parameters
node_attr_size = 6     # ini samain dengan node attr size yang udh ada one hot encoding di simulator.py
edge_attr_size = 1      # Matches dummy edge features
latent_dim_size = 18    # Latent space dimension
message_passing_num = 3 # Number of message passing steps

print_batch = 1
save_batch = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
model = Simulator(latent_dim_size=latent_dim_size, message_passing_num=message_passing_num, node_input_size=node_attr_size, edge_input_size=edge_attr_size, device=device)
optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    dataset = DatasetFormatter(max_epochs=10, dataset_dir="dataset_split")
    train_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)

    for batch_index, graph in enumerate(train_loader):
        print(f"Batch: {batch_index}")
        graph = graph.cuda()
        node_type = graph.x[:, 0]
        velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std)
        predicted_acc, target_acc = model(graph, velocity_sequence_noise) # only taking the velocity attributes
        mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.FARFIELD)

        errors = ((predicted_acc - target_acc)**2) # for x and y vel without solid body
        loss = torch.mean(errors[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print('batch %d [loss %.2e]'%(batch_index, loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint()