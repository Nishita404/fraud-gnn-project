import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):

    def __init__(self, input_dim):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(input_dim, 256)
        self.conv2 = SAGEConv(256, 128)
        self.conv3 = SAGEConv(128, 64)       # new third layer

        self.linear = torch.nn.Linear(64, 2)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv3(x, edge_index)        # new third layer
        x = F.relu(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=1)