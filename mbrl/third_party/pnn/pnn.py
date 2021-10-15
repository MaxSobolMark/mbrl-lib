import torch
import torch.nn as nn
import torch.nn.functional as F


class PNNColumn(nn.Module):
    def __init__(self, n_tasks, input_size, hidden_size, output_size, device):
        super(PNNColumn, self).__init__()
        print('device is ', device)
        self.device = device
        self.n_tasks = n_tasks
        self.sizes = [hidden_size, output_size]
        self.hidden_size = hidden_size
        self.n_layers = 3
        self.w = nn.ModuleList()
        self.u = nn.ModuleList()
        self.v = nn.ModuleList()
        self.alpha = nn.ModuleList()
        self.w.append(nn.Linear(input_size, hidden_size))
        self.w.append(nn.Linear(hidden_size, hidden_size))
        self.w.append(nn.Linear(hidden_size, output_size))
        for i in range(self.n_tasks):
            self.v.append(nn.ModuleList())
            self.v[i].append(nn.Linear(hidden_size, hidden_size))
            self.v[i].append(nn.Linear(hidden_size, hidden_size))
            self.u.append(nn.ModuleList())
            self.u[i].append(nn.Linear(hidden_size, hidden_size))
            self.u[i].append(nn.Linear(hidden_size, output_size))
            self.alpha.append(nn.ParameterList())
            self.alpha[i].append(nn.Parameter(torch.tensor(1e-2)))
            self.alpha[i].append(nn.Parameter(torch.tensor(1e-2)))
        self._reset_parameters()

    def forward(self, x, outputs):
        batch_size = x.shape[0]
        if self.n_tasks == 0:
            outputs = torch.zeros(1, self.n_layers-1, batch_size, self.hidden_size)
        else:
            outputs = torch.cat((outputs, torch.zeros(1, self.n_layers-1, batch_size, self.hidden_size)))
        y = self.w[0](x)
        outputs[self.n_tasks][0] = y.detach()
        for layer in range(self.n_layers-1):
            u_out = torch.zeros(batch_size, self.sizes[layer], device=self.device)
            for k in range(self.n_tasks):
                v_out = self.v[k][layer](self.alpha[k][layer] * (outputs[k][layer]))
                u_out += self.u[k][layer](F.relu(v_out))
            y = F.relu(u_out + self.w[layer+1](y))
            if layer != (self.n_layers-2):
                outputs[self.n_tasks][layer + 1] = y.detach()
        return y, outputs

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
        outputs = None
        for i in range(self.n_tasks):
            y, outputs = self.columns[i].forward(x, outputs)
        return y

    def parameters(self):
        return self.columns[self.n_tasks-1].parameters()


if __name__ == "__main__":

    pnn_input_size = 5
    pnn_hidden_size = 10
    pnn_output_size = 2

    pnn = PNN(pnn_input_size, pnn_hidden_size, pnn_output_size)
    pnn.new_task()
    pnn.new_task()
    pnn.new_task()

    bsize = 5
    sample_x = torch.rand(bsize, pnn_input_size)
    print(pnn.forward(sample_x))
    # for p in pnn.parameters():
    #    print(p)



