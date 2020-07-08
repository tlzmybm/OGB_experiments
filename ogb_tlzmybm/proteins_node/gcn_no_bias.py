import argparse
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot, zeros
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger

parser = argparse.ArgumentParser(description='OGBN-Proteins (Full-Batch)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--eval_steps', type=int, default=5)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-proteins')
split_idx = dataset.get_idx_split()
data = dataset[0]

edge_index = data.edge_index.to(device)
adj = SparseTensor(row=edge_index[0], col=edge_index[1]).set_diag()
adj_0 = adj
deg = adj.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)


class l_GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(l_GCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        return adj @ x @ self.weight# + self.bias


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(l_GCN(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(l_GCN(hidden_channels, hidden_channels))
        self.convs.append(l_GCN(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x):
        for conv in self.convs[:-1]:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x


class l_SAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(l_SAGE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.root_weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.root_weight)
        zeros(self.bias)

    def forward(self, x):
        out = adj_0.matmul(x, reduce="mean") @ self.weight
        out = out + x @ self.root_weight + self.bias
        return out


class l_P(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(l_P, self).__init__()

    def forward(self, x):
        return adj_0.matmul(x, reduce="mean")


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(l_SAGE(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(l_SAGE(hidden_channels, hidden_channels))
        self.convs.append(l_SAGE(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x):
        for conv in self.convs[:-1]:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x


class SAGE_Re(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Re, self).__init__()

        self.rezero = l_GCN(in_channels, in_channels)
        self.alpha = Parameter(torch.Tensor([0]))
#        self.alpha=0

        self.convs = torch.nn.ModuleList()
        self.convs.append(l_SAGE(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(l_SAGE(hidden_channels, hidden_channels))
        self.convs.append(l_SAGE(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        zeros(self.alpha)
        self.rezero.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x):
        x = x+self.alpha*self.rezero(x)
        for conv in self.convs[:-1]:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x


class GNN_C(torch.nn.Module):
    def __init__(self, in_channels, channels, num, dropout):
        super(GNN_C, self).__init__()

        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Sequential(l_GCN(in_channels, channels[0]), torch.nn.ReLU(
            True), torch.nn.Dropout(p=self.dropout)))
        for _ in range(1, num):
            self.convs.append(torch.nn.Sequential(l_GCN(channels[_-1], channels[_]), torch.nn.ReLU(
                True), torch.nn.Dropout(p=self.dropout), l_P(channels[_], channels[_])))
        self.convs.append(torch.nn.Sequential(
            l_GCN(channels[num-1], channels[num])))

        self.out_weights = []
        self.out_bias = []
        for i in range(1):
            self.out_weights.append(
                Parameter(torch.Tensor(channels[num+i], channels[num+i+1])).to(device))
            self.out_bias.append(
                Parameter(torch.Tensor(channels[num+i+1])).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv[0].reset_parameters()
        for i in range(1):
            glorot(self.out_weights[i])
            zeros(self.out_bias[i])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = x @ self.out_weights[0] + self.out_bias[0]
        return x


def train(model, x, y_true, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(x)[train_idx]
    loss = criterion(out, y_true[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator):
    model.eval()

    y_pred = model(x)

    train_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


if __name__ == "__main__":

    x = scatter(data.edge_attr, data.edge_index[0], dim=0,
                dim_size=data.num_nodes, reduce='mean').to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    # if args.use_sage:
    # model = SAGE(x.size(-1), args.hidden_channels, 112,
    # args.num_layers, args.dropout).to(device)
    # else:
    # model = SAGE_Re(x.size(-1), args.hidden_channels, 112,
    # args.num_layers, args.dropout).to(device)
    model = GCN(x.size(-1), args.hidden_channels, 112,
                args.num_layers, args.dropout).to(device)

#    channels = [1024, 512, 256, 256, 112]
#    model = GNN_C(x.size(-1), channels, 3, args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, x, y_true, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()
