import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PNNColumn(nn.Module):
    def __init__(self, n_tasks, input_size, hidden_size, output_size, device):
        super(PNNColumn, self).__init__()
        print('device is ', device)
        print('current task id is ', n_tasks)
        self.device = device
        self.n_tasks = n_tasks
        self.sizes = [hidden_size, output_size]
        self.hidden_size = hidden_size
        self.n_layers = 3
        self.w = nn.ModuleList()
        self.w.append(nn.Linear(input_size, hidden_size))
        self.w.append(nn.Linear(hidden_size, hidden_size))
        self.w.append(nn.Linear(hidden_size, output_size))
        self.w = self.w.to(self.device)
        if self.n_tasks:
            self.u = nn.ModuleList()
            self.u.append(nn.Linear(hidden_size, hidden_size))
            self.u.append(nn.Linear(hidden_size, output_size))
            self.v = nn.ModuleList()
            self.v.append(nn.Linear(hidden_size*self.n_tasks, hidden_size))
            self.v.append(nn.Linear(hidden_size*self.n_tasks, hidden_size))
            self.alpha = nn.ModuleList()
            for i in range(self.n_tasks):
                self.alpha.append(nn.ParameterList())
                self.alpha[i].append(nn.Parameter(torch.tensor(1e-3)))
                self.alpha[i].append(nn.Parameter(torch.tensor(1e-3)))
            self.u = self.u.to(self.device)
            self.v = self.v.to(self.device)
            self.alpha = self.alpha.to(self.device)
        self._reset_parameters()

    def forward(self, x, prev_out):
        batch_size = x.shape[0]
        out = torch.zeros(self.n_layers-1, batch_size, self.hidden_size).to(self.device).detach()
        y = F.relu(self.w[0](x))
        out[0] = y.detach()
        for layer in range(self.n_layers-1):
            y = self.w[layer + 1](y)
            if self.n_tasks:
                v_in = torch.zeros(batch_size, self.hidden_size*self.n_tasks).to(self.device)
                for k in range(self.n_tasks):
                    v_in[:, k*self.hidden_size:(k+1)*self.hidden_size] = self.alpha[k][layer] * prev_out[k][layer]
                u_out = self.u[layer](F.relu(self.v[layer](v_in)))
                y = y + u_out
            if layer != (self.n_layers-2):
                y = F.relu(y)
                out[layer + 1] = y.detach()
        return y, out

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class PNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device='cpu'):
        super(PNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.columns = nn.ModuleList()
        self.n_tasks = 0

    def new_task(self):
        for i in range(self.n_tasks):
            self.columns[i].freeze()
        new_col = PNNColumn(self.n_tasks, self.input_size, self.hidden_size,
                            self.output_size, self.device).to(self.device)
        self.columns.append(new_col)
        self.n_tasks += 1

    def forward(self, x):
        outputs = []
        for i in range(self.n_tasks):
            y, out = self.columns[i].forward(x, outputs)
            outputs.append(out)
        return y

    def parameters(self):
        return self.columns[self.n_tasks-1].parameters()


if __name__ == "__main__":
    device = 'cpu'  # 'cuda:0'
    pnn_input_size = 1
    pnn_hidden_size = 10
    pnn_output_size = 2
    # preparing the data
    data_x = torch.arange(-1, 1, 0.1)
    ndata = data_x.shape[0]
    data_y = torch.zeros(ndata, 2)
    data_y[:, 0] = torch.sin(data_x)
    data_y[:, 1] = torch.cos(data_x)
    data_x = data_x.view(ndata, 1)
    torch.autograd.set_detect_anomaly(True)
    pnn = PNN(pnn_input_size, pnn_hidden_size, pnn_output_size, device)
    pnn.new_task()
    pnn.new_task()
    # setting the optimizer
    optimizer = optim.Adam(pnn.parameters(), 0.001)
    # training
    for _ in range(10):
        optimizer.zero_grad()
        yhat = pnn.forward(data_x)
        loss = F.mse_loss(data_y, yhat)
        loss.backward()
        optimizer.step()




