import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot, zeros, ones

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger

parser = argparse.ArgumentParser(description='OGBL-PPA (Full-Batch)')
parser.add_argument('--decode', action='store_true')
parser.add_argument('--load', action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--channel', type=int, default=64)
parser.add_argument('--use_node_embedding', action='store_true')
parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=64 * 1024)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygLinkPropPredDataset(name='ogbl-ppa')
data = dataset[0]
split_edge = dataset.get_edge_split()

edge_index = data.edge_index.to(device)
adj_0 = SparseTensor(row=edge_index[0], col=edge_index[1])

# Pre-compute GCN normalization.
adj = adj_0.set_diag()
deg = adj.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)


class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        return adj @ x @ self.weight + self.bias


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

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


class SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__()

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
        out = adj_0.matmul(x, reduce='mean') @ self.weight
        out = out + x @ self.root_weight + self.bias
        return out


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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


class MLP(torch.nn.Module):
    def __init__(self, in_channels, dropout):
        super(MLP, self).__init__()

        self.dropout = dropout
        self.channels = [args.channel, 4*args.channel, args.channel]
        self.MLP = torch.nn.ModuleList()
        self.alpha = Parameter(torch.Tensor([0]))
        for i in range(0, 2):
            self.MLP.append(torch.nn.Sequential(torch.nn.Linear(
                self.channels[i], self.channels[i+1]), torch.nn.Tanh(), torch.nn.Dropout(p=self.dropout)))

    def reset_parameters(self):
        zeros(self.alpha)
        for layer in self.MLP:
            layer[0].reset_parameters()

    def forward(self, x0):
        x = x0
        for layer in self.MLP:
            x = layer(x)
        return x * self.alpha + x0


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LP_dot(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LP_dot, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        n = x_i.size()[0]
        x = torch.bmm(x_i.view(n, 1, args.channel),
                      x_j.view(n, args.channel, 1))
        return torch.sigmoid(x)


def train(model, predictor, x, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):

        optimizer.zero_grad()

        h = model(x)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long,
                             device=x.device)

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, split_edge, evaluator, batch_size):
    model.eval()

    h = model(x)

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


if __name__ == "__main__":

    if args.use_node_embedding:
        x = data.x.to(torch.float)
        x = torch.cat([x, torch.load('embedding.pt')], dim=-1)
        x = x.to(device)
    else:
        x = data.x.to(torch.float).to(device)

    print(data.num_nodes)

    if args.decode:
        if args.load:
            x = torch.load('x.pt', map_location=device)
            print(torch.max(x, 0))
        else:
            W = SparseTensor(row=data.edge_index[1], col=data.edge_index[0]).to(
                device).set_diag()
            x = torch.normal(mean=0, std=1, size=(
                data.num_nodes, args.channel)).to(device)
            for j in range(300):
                print(j)
                print(x)
                x = W @ x
                x, r = torch.qr(x)
            x = F.normalize(x, 1)
        torch.save(x, 'x.pt')
        model = MLP(x.size(-1), args.dropout).to(device)
        predictor = LP_dot(args.channel, args.hidden_channels,
                           1, args.num_layers, args.dropout).to(device)
    else:
        if args.use_sage:
            model = SAGE(x.size(-1), args.hidden_channels, args.hidden_channels,
                         args.num_layers, args.dropout).to(device)
        else:
            model = GCN(x.size(-1), args.hidden_channels, args.hidden_channels,
                        args.num_layers, args.dropout).to(device)

        predictor = LinkPredictor(
            args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ppa')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, x, split_edge, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, x, split_edge, evaluator,
                               args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
