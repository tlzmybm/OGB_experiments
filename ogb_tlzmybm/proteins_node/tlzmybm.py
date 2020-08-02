import argparse
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot, zeros, ones
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger

parser = argparse.ArgumentParser(description='OGBN-Proteins (Full-Batch)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--lr', type=float, default=0.01)
#parser.add_argument('--alpha', type=float, default=0.01)
# parser.add_argument('--mt', type=float, default=0.72)
parser.add_argument('--weight_decay', type=float, default=0.007)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--eval_steps', type=int, default=5)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--decode', action='store_true')
parser.add_argument('--load', action='store_true')
parser.add_argument('--channel', type=int, default=64)
args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-proteins')
split_idx = dataset.get_idx_split()
data = dataset[0]

edge_index = data.edge_index.to(device)
adj_0 = SparseTensor(row=edge_index[0], col=edge_index[1])
adj = adj_0.set_diag()
deg = adj.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
lamb = 0.7
loss_tmp = torch.Tensor([0]).to(device)


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
        return adj @ x @ self.weight + self.bias


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
    def __init__(self, in_channels, dropout):
        super(SAGE_Re, self).__init__()

        self.channels = [in_channels, in_channels, 256, 256, 256, 112]
        self.num_layers = 5
        self.alpha = Parameter(torch.Tensor(self.num_layers))
        zeros(self.alpha)
        self.zero = l_GCN(in_channels, in_channels)
        self.rezero = [1, 0, 0, 1, 0]
        self.relu_list = [1, 2]
        self.re_list = [0, 3]

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i in self.re_list:
                self.convs.append(l_GCN(self.channels[i], self.channels[i+1]))
            else:
                self.convs.append(l_SAGE(self.channels[i], self.channels[i+1]))

        self.dropout = dropout

    def reset_parameters(self):
        zeros(self.alpha)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x):
        y = [x]
        for i in range(self.num_layers):
            conv = self.convs[i]
            tmp = conv(x)
            if i in self.relu_list:
                tmp = F.relu(tmp)
            if i in self.re_list:
                x = y[i+1-self.rezero[i]]+self.alpha[i]*tmp
            else:
                x = tmp
            x = F.dropout(x, p=self.dropout, training=self.training)
            y.append(x)
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


class MLP(torch.nn.Module):
    def __init__(self, in_channels, dropout):
        super(MLP, self).__init__()

        self.dropout = dropout
        self.channels = [in_channels, 512, 256, 112]
        self.MLP_co = torch.nn.ModuleList()
        self.alpha = Parameter(torch.Tensor([0]))
        self.beta = Parameter(torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1]))
        for i in range(8):
            self.MLP_co.append(torch.nn.Sequential(
                torch.nn.Linear(args.channel, self.channels[1]//8), torch.nn.Tanh()))
        self.MLP = torch.nn.ModuleList()
        self.MLP1 = torch.nn.ModuleList()
        self.MLP2 = torch.nn.ModuleList()
        self.MLP1.append(torch.nn.Sequential(torch.nn.Linear(
            56, self.channels[1]), torch.nn.ReLU(True), torch.nn.Dropout(p=self.dropout)))
        for i in range(1, 2):
            self.MLP1.append(torch.nn.Sequential(torch.nn.Linear(
                self.channels[i], self.channels[i+1]), torch.nn.ReLU(True), torch.nn.Dropout(p=self.dropout)))
            self.MLP2.append(torch.nn.Sequential(torch.nn.Linear(
                self.channels[i], self.channels[i+1]), torch.nn.Tanh(), torch.nn.Dropout(p=self.dropout)))
        self.MLP.append(torch.nn.Sequential(
            torch.nn.Linear(2*self.channels[2], self.channels[3])))

    def reset_parameters(self):
        ones(self.beta)
        zeros(self.alpha)
        for layer in self.MLP_co:
            layer[0].reset_parameters()
        for layer in self.MLP:
            layer[0].reset_parameters()
        for layer in self.MLP1:
            layer[0].reset_parameters()
        for layer in self.MLP2:
            layer[0].reset_parameters()

    def forward(self, x0):
        global loss_tmp
        loss_tmp = torch.Tensor([0]).to(device)
        loss_tmp.requires_grad = True
        x = x0[:, :56]
        x_ = x0[:, 56:]
        for layer in self.MLP1:
            x = layer(x)
        y = self.beta[0] * \
            self.MLP_co[0](x_[:, :args.channel])
        for i in range(1, 8):
            y = torch.cat(
                (y, self.beta[i]*self.MLP_co[i](x_[:, i*args.channel:(i+1)*args.channel])), 1)
            loss_tmp = loss_tmp + torch.abs(self.beta[i]) * \
                torch.norm(self.MLP_co[i][0].weight, p=2)
        for layer in self.MLP2:
            y = layer(y)
            loss_tmp = loss_tmp + torch.norm(layer[0].weight, p=2)
        x = torch.cat((x, y * self.alpha), 1)
#        x = x
        for layer in self.MLP:
            x = layer(x)
            loss_tmp = loss_tmp+torch.norm(layer[0].weight, p=2)
        return x


def train(model, x, y_true, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(x)[train_idx]
    loss = criterion(out, y_true[train_idx].to(torch.float))
#    if loss < 0.29:
#        zeros(model.alpha)
#        model.alpha.requires_grad = False
#        if abs(model.alpha) > 1e-7:
#            print(model.alpha)
#    else:
#        model.alpha.requires_grad = True
    if loss < 0:
        print(loss)
    loss = loss + torch.abs(model.alpha) * loss_tmp * args.weight_decay
    if loss_tmp < 0:
        print(loss_tmp)
    loss.backward()
    optimizer.step()

    return loss.item()


@ torch.no_grad()
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

    x_a = data.edge_attr
    tmp = x_a
    x = scatter(tmp.to(device), edge_index[0], dim=0,
                dim_size=data.num_nodes, reduce='mean')
    for i in range(6):
        tmp = tmp*x_a
        x = torch.cat((x, scatter(
            tmp.to(device), edge_index[0], dim=0, dim_size=data.num_nodes, reduce='mean')), 1)

    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)
    if args.decode:
        if args.load:
            x = torch.load('x.pt', map_location=device)
            x_ = x[:, :56]
            x__ = x[:, 56:]
            y = F.normalize(x__[:, :args.channel], p=2, dim=1)
            for i in range(1, 8):
                y = torch.cat(
                    (y, F.normalize(x__[:, i*args.channel:(i+1)*args.channel], p=2, dim=1)), 1)
            x = torch.cat((x_, y), 1)
            print(torch.max(x, 0))
        else:
            for i in range(8):
                W = SparseTensor(
                    row=data.edge_index[1], col=data.edge_index[0], value=x_a[:, i]).to(device).set_diag()
                tmp = torch.normal(mean=0, std=1, size=(
                    data.num_nodes, args.channel)).to(device)
                print(i)
                for j in range(0):
                    print(i)
                    print(j)
                    print(tmp)
                    tmp = W @ tmp
                    tmp, r = torch.qr(tmp)
#                tmp = W @ tmp
#                 tmp = torch.load('tmp.pt')
                # torch.save(tmp, 'tmp.pt')
                # print(tmp)
                # print(torch.max(tmp, 0))
#                tmp = 721/(torch.max(tmp)-torch.min(tmp))*tmp
                # print(tmp)
                # print(torch.max(tmp, 0))
                x = torch.cat((x, tmp), 1)
                # print(torch.max(x, 0))
#            torch.save(x, 'x.pt')
        model = MLP(x.size(-1), args.dropout).to(device)
    else:
        if args.use_sage:
            model = SAGE(x.size(-1), args.hidden_channels, 112,
                         args.num_layers, args.dropout).to(device)
        else:
            model = SAGE_Re(x.size(-1), args.dropout).to(device)
#    model = GCN(x.size(-1), args.hidden_channels, 112,
#                args.num_layers, args.dropout).to(device)

#    channels = [1024, 512, 256, 256, 112]
#    model = GNN_C(x.size(-1), channels, 3, args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.99, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, x, y_true, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    print(model.alpha)
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()
