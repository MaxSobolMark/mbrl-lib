import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PNNColumn(nn.Module):
    def __init__(self, n_tasks, input_size, hidden_size, pnn_hidden_size, output_size, device):
        super(PNNColumn, self).__init__()
        print('device is ', device)
        print('current task id is ', n_tasks)
        self.device = device
        self.n_tasks = n_tasks
        self.hidden_size = hidden_size
        self.pnn_hidden_size = pnn_hidden_size
        self.n_layers = 3
        self.w = nn.ModuleList()
        self.w.append(nn.Linear(input_size, hidden_size))
        self.w.append(nn.Linear(hidden_size, hidden_size))
        self.w.append(nn.Linear(hidden_size, output_size))
        self.w = self.w.to(self.device)

        if self.n_tasks:
            self.u = nn.ModuleList()
            self.u.append(nn.Linear(self.pnn_hidden_size, hidden_size))
            self.u.append(nn.Linear(self.pnn_hidden_size, output_size))
            self.v = nn.ModuleList()
            self.v.append(nn.Linear(hidden_size*self.n_tasks, self.pnn_hidden_size))
            self.v.append(nn.Linear(hidden_size*self.n_tasks, self.pnn_hidden_size))
            self.alpha = nn.ModuleList()
            for i in range(self.n_tasks):
                self.alpha.append(nn.ParameterList())
                self.alpha[i].append(nn.Parameter(torch.tensor(1e-10)))
                self.alpha[i].append(nn.Parameter(torch.tensor(1e-10)))
            self.u = self.u.to(self.device)
            self.v = self.v.to(self.device)
            self.alpha = self.alpha.to(self.device)
        self._reset_parameters()

    def forward(self, x, prev_out):
        batch_size = x.shape[0]
        out = torch.zeros(self.n_layers-1, batch_size, self.hidden_size).to(self.device)
        y = F.relu(self.w[0](x))
        out[0] = y
        for layer in range(self.n_layers-1):
            y = self.w[layer + 1](y)
            if self.n_tasks:
                v_in = torch.zeros(batch_size, self.hidden_size*self.n_tasks).to(self.device)
                for k in range(self.n_tasks):
                    v_in[:, k*self.hidden_size:(k+1)*self.hidden_size] = self.alpha[k][layer] \
                                                                         * prev_out[k][layer]
                u_out = self.u[layer](F.relu(self.v[layer](v_in)))
                y = y + u_out
            if layer != (self.n_layers-2):
                y = F.relu(y)
                out[layer + 1] = y
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
    def __init__(self, input_size, hidden_size, pnn_hidden_size, output_size, device='cpu'):
        super(PNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.pnn_hidden_size = pnn_hidden_size
        self.columns = nn.ModuleList()
        self.n_tasks = 0

    def new_task(self):
        for i in range(self.n_tasks):
            self.columns[i].freeze()
        new_col = PNNColumn(self.n_tasks, self.input_size, self.hidden_size,
                            self.pnn_hidden_size, self.output_size, self.device).to(self.device)
        self.columns.append(new_col)
        self.n_tasks += 1

    def forward(self, x, task_id=-1):
        task_id = self.n_tasks if task_id < 0 else task_id
        outputs = []
        for i in range(task_id):
            #if i == 2:
                #x += torch.rand(x.shape)
            #    print ('Yes ', task_id)
            y, out = self.columns[i].forward(x, outputs)
            outputs.append(out)
        return y

    def parameters(self):
        return self.columns[self.n_tasks-1].parameters()


if __name__ == "__main__":
    device = 'cpu'  # 'cuda:0'
    num_tasks = 1
    num_repeats = 3

    torch.autograd.set_detect_anomaly(True)
    all_train_loss = np.zeros((1, num_tasks * num_repeats * 300 ))
    ndata = 50
    for ctr in range(1):
        # preparing the data
        data_x = torch.arange(-1, 1, 0.1)  #torch.rand(ndata) * 2 - 1
        ndata = data_x.shape[0]
        data_y = torch.zeros(num_tasks, ndata, 2)
        data_y[0, :, 0] = torch.sin(data_x) + torch.pow(data_x, 3) - torch.pow(data_x, 2)
        data_y[0, :, 1] = torch.cos(data_x) + torch.pow(torch.cos(data_x),2) + torch.pow(data_x, 3) - torch.pow(data_x, 2)
        #data_y[1, :, 0] = torch.sin(data_x)
        #data_y[1, :, 1] = torch.cos(data_x)
        #data_y[2, :, 0] = torch.sin(data_x)
        #data_y[2, :, 1] = torch.cos(data_x)
        data_x = data_x.view(ndata, 1)

        pnn = PNN(1, 10, 3, 2, device)
        train_loss = []
        for _ in range(num_repeats):
            for task in range(num_tasks):
                pnn.new_task()
                # setting the optimizer
                optimizer = optim.Adam(pnn.parameters(), 0.001)
                # training
                sum_loss = 0
                for epoch in range(3000):
                    optimizer.zero_grad()
                    yhat = pnn.forward(data_x)
                    loss = F.mse_loss(data_y[task], yhat)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item()
                    if epoch % 10 == 9:
                        train_loss = np.append(train_loss, sum_loss/10)
                        sum_loss = 0
        all_train_loss[ctr] = train_loss

    mean = np.mean(all_train_loss, axis=0)
    standard_dev = np.std(all_train_loss, axis=0)

    import matplotlib.pyplot as plt
    plt.fill_between(np.arange(train_loss.shape[0]), mean - standard_dev, mean + standard_dev)
    plt.plot(np.arange(train_loss.shape[0]), mean, 'r-')

    yhat = pnn.forward(data_x, 1)
    loss = F.mse_loss(data_y[0], yhat)
    print('task {:d}, loss {:f}'.format(0, loss.item()))

    yhat = pnn.forward(data_x, 2)
    loss = F.mse_loss(data_y[0], yhat)
    print('task {:d}, loss {:f}'.format(1, loss.item()))

    yhat = pnn.forward(data_x, 3)
    loss = F.mse_loss(data_y[0], yhat)
    print('task {:d}, loss {:f}'.format(2, loss.item()))

    yhat = pnn.forward(data_x, 1)
    loss = F.mse_loss(data_y[0], yhat)
    print('task {:d}, loss {:f}'.format(0, loss.item()))

    yhat = pnn.forward(data_x, 2)
    loss = F.mse_loss(data_y[0], yhat)
    print('task {:d}, loss {:f}'.format(1, loss.item()))

    yhat = pnn.forward(data_x, 3)
    loss = F.mse_loss(data_y[0], yhat)
    print('task {:d}, loss {:f}'.format(2, loss.item()))

    pnn.columns[0]._reset_parameters()

    yhat = pnn.forward(data_x, 2)
    loss = F.mse_loss(data_y[0], yhat)
    print('task {:d}, loss {:f}'.format(1, loss.item()))

    yhat = pnn.forward(data_x, 3)
    loss = F.mse_loss(data_y[0], yhat)
    print('task {:d}, loss {:f}'.format(2, loss.item()))

    plt.show()








