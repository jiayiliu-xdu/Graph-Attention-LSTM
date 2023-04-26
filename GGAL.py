# Import related dependent libraries
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import timedelta, date
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.data import Data
import torch_geometric.transforms as T
from scipy.signal import savgol_filter

# Set related parameters
TEST_PERCENT = 0.2
# N: Using previous N to predict the next
N = 24
# O: Output feature size before the GAT’s last layer 
O = 16
# H: Attention mechanism number (heads)
H = 12
# L: Number of hidden feature size in LSTM
L = 64

# Dataset import and preprocessing
df0 = pd.read_csv('achieve/bono-load.avg_1_min.csv', delimiter=',')
df0.dataframeName = 'bono-load.avg_1_min.csv'
df0['Timestamp'],df0['value']= df0['Timestamp;value'].str.split(';',1).str
df0 = df0.drop('Timestamp;value',axis = 1)
df0.rename(columns={'value':'Load bono'},inplace = True)
f0 = savgol_filter(df0['Load bono'], 9, 5, mode= 'nearest')
f0 = pd.DataFrame(f0)
f0.columns = {'Load bono'}
df0 = df0.drop('Load bono',axis = 1)
fz0 = pd.concat([df0,f0],axis=1)

df1 = pd.read_csv('achieve/ellis-load.avg_1_min.csv', delimiter=',')
df1.dataframeName = 'ellis-load.avg_1_min.csv'
df1['Timestamp'],df1['value']= df1['Timestamp;value'].str.split(';',1).str
df1 = df1.drop('Timestamp;value',axis = 1)
df1 = df1.drop('Timestamp',axis = 1)
df1.rename(columns={'value':'Load ellis'},inplace = True)
f1 = savgol_filter(df1['Load ellis'], 9, 5, mode= 'nearest')
f1 = pd.DataFrame(f1)
f1.columns = {'Load ellis'}
df1 = df1.drop('Load ellis',axis = 1)
fz1 = pd.concat([df1,f1],axis=1)

df2 = pd.read_csv('achieve/homer-load.avg_1_min.csv', delimiter=',')
df2.dataframeName = 'homer-load.avg_1_min.csv'
df2['Timestamp'],df2['value']= df2['Timestamp;value'].str.split(';',1).str
df2 = df2.drop('Timestamp;value',axis = 1)
df2 = df2.drop('Timestamp',axis = 1)
df2.rename(columns={'value':'Load homer'},inplace = True)
df2 = df2.drop(177096)
df2.rename(columns={'value':'Load homer'},inplace = True)
f2 = savgol_filter(df2['Load homer'], 9, 4, mode= 'nearest')
f2 = pd.DataFrame(f2)
f2.columns = {'Load homer'}
df2 = df2.drop('Load homer',axis = 1)
fz2 = pd.concat([df2,f2],axis=1)

df3 = pd.read_csv('achieve/homestead-load.avg_1_min.csv', delimiter=',')
df3['Timestamp'],df3['value']= df3['Timestamp;value'].str.split(';',1).str
df3 = df3.drop('Timestamp;value',axis = 1)
df3 = df3.drop('Timestamp',axis = 1)
df3.rename(columns={'value':'Load homestead'},inplace = True)
df3 = df3.drop(177096)
df3.rename(columns={'value':'Load homestead'},inplace = True)
f3 = savgol_filter(df3['Load homestead'], 9, 7, mode= 'nearest')
f3 = pd.DataFrame(f3)
f3.columns = {'Load homestead'}
df3 = df3.drop('Load homestead',axis = 1)
fz3 = pd.concat([df3,f3],axis=1)

df4 = pd.read_csv('achieve/ralf-load.avg_1_min.csv', delimiter=',')
df4['Timestamp'],df4['value']= df4['Timestamp;value'].str.split(';',1).str
df4 = df4.drop('Timestamp;value',axis = 1)
df4 = df4.drop('Timestamp',axis = 1)
df4.rename(columns={'value':'Load ralf'},inplace = True)
df4.rename(columns={'value':'Load ralf'},inplace = True)
f4 = savgol_filter(df4['Load ralf'], 13, 5, mode= 'nearest')
f4 = pd.DataFrame(f4)
f4.columns = {'Load ralf'}
df4 = df4.drop('Load ralf',axis = 1)
fz4 = pd.concat([df4,f4],axis=1)

df5 = pd.read_csv('achieve/sprout-load.avg_1_min.csv', delimiter=',')
df5['Timestamp'],df5['value']= df5['Timestamp;value'].str.split(';',1).str
df5 = df5.drop('Timestamp;value',axis = 1)
df5 = df5.drop('Timestamp',axis = 1)
df5.rename(columns={'value':'Load sprout'},inplace = True)
f5 = savgol_filter(df5['Load sprout'], 9, 7, mode= 'nearest')
f5 = pd.DataFrame(f5)
fz = pd.concat([df5,f5],axis=1)
fz = fz.drop('Load sprout',axis = 1)
fz.columns = {'Load sprout'}

import matplotlib.dates as mdates
fz0['Load bono'] = fz0['Load bono'].astype(float)
fz1['Load ellis'] = fz1['Load ellis'].astype(float)
fz2['Load homer'] = fz2['Load homer'].astype(float)
fz3['Load homestead'] = fz3['Load homestead'].astype(float)
fz4['Load ralf'] = fz4['Load ralf'].astype(float)
df5['Load sprout'] = df5['Load sprout'].astype(float)
add_df=pd.concat([fz0,fz1,fz2,fz3,fz4,fz],axis=1)

add_df = add_df.drop(177096)
add_df = add_df.iloc[10047:177095,:]
add_df = add_df.iloc[0:37452,:]
add_df = add_df.reset_index(drop=True)

from datetime import datetime
add_df['Timestamp'] = pd.to_datetime(add_df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

scaler = MinMaxScaler(feature_range=(0, 1))
add_df['Load bono'] = scaler.fit_transform(add_df['Load bono'].values.reshape(-1, 1))
add_df['Load ellis'] = scaler.fit_transform(add_df['Load ellis'].values.reshape(-1, 1))
add_df['Load homer'] = scaler.fit_transform(add_df['Load homer'].values.reshape(-1, 1))
add_df['Load homestead'] = scaler.fit_transform(add_df['Load homestead'].values.reshape(-1, 1))
add_df['Load ralf'] = scaler.fit_transform(add_df['Load ralf'].values.reshape(-1, 1))
add_df['Load sprout'] = scaler.fit_transform(add_df['Load sprout'].values.reshape(-1, 1))

# Import generated dataset by using TIMEGAN and preprocessing
df6 = pd.read_csv('achieve/loadsynd.csv')
df6 = df6.drop('Unnamed: 0',axis = 1)
df6.rename(columns={'0':'Load fw'},inplace = True)
df6.rename(columns={'1':'Load lb'},inplace = True)
df6 = df6.iloc[0:37452,:]

from scipy.signal import savgol_filter
x = savgol_filter(df6['Load fw'], 11, 7, mode= 'nearest')
y = savgol_filter(df6['Load lb'], 15, 11, mode= 'nearest')
x = pd.DataFrame(x)
y = pd.DataFrame(y)
dfz = pd.concat([df6,x],axis=1)
dfz = pd.concat([dfz,y],axis=1)
dfz = dfz.drop('Load fw',axis = 1)
dfz = dfz.drop('Load lb',axis = 1)
dfz.columns = {'Load fw','Load lb'}

add_df = pd.concat([add_df,dfz],axis=1)

# Split the training set and test set
used_data = add_df
test_size = int(used_data.shape[0] * TEST_PERCENT)                                   
train_data = used_data.iloc[:-test_size, :].copy()
train_data = train_data.reset_index(drop=True)
test_data = used_data.iloc[-test_size-N:, :].copy()
test_data = test_data.reset_index(drop=True)

# Create the input to the Model
def create_input_graph_set(data, nodes_num):

    # return the a set of graphs for GAT

    result_graphs = []
    features_num = int((data.shape[1] - 1) // nodes_num)
    # connect all nodes with the other and it self
    
    
    edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6],
                               [4, 5, 2, 3, 7, 1, 5, 1, 5, 0, 5, 0, 2, 3, 4, 0]], dtype=torch.long)
    
    for r in range (data.shape[0]):
        row = list(data.iloc[r])
        features = []
        for f in range(int((len(row)-1) / features_num)):
            features.append(row[1 + f * features_num : 1 + (f + 1) * features_num])
        x = torch.tensor(features, dtype=torch.float)
        result_graphs.append(Data(x = x, edge_index = edge_index))
        
    return result_graphs
    
def get_all_y_and_y(data, VNF_names, training_length, target_id):

    # return all_y which is a dataframe contains y values for all assets 
    # and y which is a tensor contains y values for the target asset

    all_y = pd.DataFrame()
    for n in VNF_names:
        used_col = str("Load " + n)
        all_y[used_col] = data[used_col]
    all_y.insert(0, "Date", data['Timestamp'][1:])
    
    y = all_y.iloc[training_length : , target_id+1:target_id+2]
    return all_y, torch.tensor(y.values).float()

# Rewrite GATConv from torch_geometric
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


class GATConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        s = alpha.clone()
        v = alpha.clone()
        for x in range(s.shape[1]):         # Assign 1 or 0 to s_{ij} related to subNF
            s[7,x] = 1
            s[14,x] = 1
            s[1,x] = 0
            s[2,x] = 0
            s[12,x] = 0
            s[19,x] = 0
            s[22,x] = 0
        for y in range(v.shape[1]):         # Assign 1 or 0 to s_{ij} related to VNF
            v[10,y] = 0.5
            v[13,y] = 0.5
            v[17,y] = 0
        alpha_new = [0.8*alpha+0.1*s+0.1*v]
        g = torch.cat(alpha_new)
        alpha = g
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

# GGAL model (GAT + LSTM + MLP)
class GATandLSTM(torch.nn.Module):
    # Stateful LSTM
    def __init__(self, graph_num, VNF_num, feature_num, target_VNF_id, training_length, lstm_hidden, device):
        super(GATandLSTM, self).__init__()
        self.device = device
        self.training_length = training_length
        self.target_id = target_VNF_id  # VNF id for prediction
        self.num_nodes = VNF_num
        # GAT
        self.gat_num = graph_num
        self.gat_features = feature_num
        self.gat_hid = O
        self.gat_head = H 
        self.gat_out = 1 
        
        
        self.gat_conv1 = GATConv(self.gat_features, self.gat_hid , heads=self.gat_head, dropout=0)
        self.gat_conv2 = GATConv(self.gat_hid*self.gat_head, self.gat_out, concat=False,
                             heads=self.gat_out, dropout=0)
        #LSTM
        self.lstm_features = self.gat_out
        self.lstm_hid = lstm_hidden
        self.lstm_seq_len = self.training_length
        self.lstm_lay = 2
        self.lstm_out = 1
        
        self.lstm = nn.LSTM(input_size=self.lstm_features, hidden_size=self.lstm_hid, 
                            num_layers=self.lstm_lay,batch_first = True, dropout=0)
        self.lin = nn.Linear(self.lstm_hid, self.lstm_out)
        
    
    def forward(self, data):
        
        # GAT forward
        for t in range(self.gat_num):
            x, edge_index = data[t].x.to(device), data[t].edge_index.to(device)
            #x = F.dropout(x, p=0.2, training=self.training)
            x = self.gat_conv1(x, edge_index)
            x = F.elu(x)
            #x = F.dropout(x, p=0.2, training=self.training)
            x = self.gat_conv2(x, edge_index)
            if t == 0:
                xs_out = x.T
            else:
                xs_out = torch.cat((xs_out, x.T), 0)
        
        # LSTM forward
        sequences = xs_out.T
        time_series = self.split_sequences(sequences, self.training_length, self.target_id)

        h0 = torch.zeros(self.lstm_lay, time_series.size(0), self.lstm_hid).to(self.device).requires_grad_()
        c0 = torch.zeros(self.lstm_lay, time_series.size(0), self.lstm_hid).to(self.device).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(time_series, (h0.detach(), c0.detach()))
        pred = self.lin(lstm_out[:, -1,:])
        
        return pred

    def split_sequences(self, sequences, training_length, target_id):
        ts_raw = sequences[target_id].detach().to('cpu').numpy()
        ts = []
        for i in range(len(ts_raw) - training_length): 
            ts.append(ts_raw[i : i + training_length])
        ts = np.array(ts)
        ts = ts.reshape(ts.shape[0],ts.shape[1], 1)
        return torch.from_numpy(ts).type(torch.Tensor).to(self.device)
    
    def set_gat_num(self, num):
        self.gat_num = num
        
# Train
def training(input_graphs, y, target_VNF_id, target_L, device):
    epoch_num = 5000
    loss_list = []
    model = GATandLSTM(len(train_data), len(names), feature_num, target_VNF_id, N, target_L, device).to(device)
    criteria = nn.MSELoss(reduction = "mean")
    
    for graph in input_graphs:
        graph = graph.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.010)
    #optimizer = torch.optim.SGD(model.parameters(), lr =0.005)

    for epoch in range(epoch_num): 
        model.train()
        out = model(input_graphs)
        loss = criteria(out, y.to(device))
        loss_list.append(loss.item())
        if epoch % 100 == 0:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, loss_list
    
# Plot function
def plot_loss(loss_list, target_VNF_id):

    # Show Loss plot

    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Loss of " + str(names[target_VNF_id]))
    plt.show()

def plot_prediction_against_truth(m, y, target_VNF_id):

    # Show Pred vs Ground Truth

    plt.plot(y.T.numpy().reshape(y.shape[0]))                    # blue line is Ground Truth
    plt.plot(m.T.detach().to('cpu').numpy().reshape(y.shape[0])) # orange line is model prediction
    plt.xlabel("Timestamp (Week)")
    plt.ylabel("Normalized Price")
    plt.title(names[target_VNF_id])
    plt.show()

names = ['bono',
         'ellis',
         'homer',
         'homestead',
         'ralf',
         'sprout',
         'fw',
         'lb']
feature_num = 1

# Main function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_graphs = create_input_graph_set(train_data, len(names))
criteria = nn.MSELoss(reduction = "mean")

train_loss, test_loss, acc_result = [], [], []

asset_id = 0
all_train_y, train_y = get_all_y_and_y(train_data, names, N, asset_id)
    
used_L= L     
m, loss_list =training(train_graphs, train_y, asset_id, used_L, device)
pred_train = m(train_graphs)
plot_loss(loss_list, asset_id)

train_loss.append(criteria(pred_train, train_y.to(device)).item())          
test_graphs = create_input_graph_set(test_data, len(names))
all_test_y, test_y= get_all_y_and_y(test_data, names, N, asset_id)
m.set_gat_num(len(test_data))

pred_test = m(test_graphs)
tl = criteria(pred_test, test_y.to(device)).item()
test_loss.append(tl)
plot_prediction_against_truth(pred_test[0:100,:], test_y[0:100,:], asset_id) 
    
validation_actual = np.sign(test_data.iloc[N-1:-1, asset_id+1:asset_id+2].copy().to_numpy() - test_y.detach().to('cpu').numpy())
validation_pred = np.sign(test_data.iloc[N-1:-1, asset_id+1:asset_id+2].copy().to_numpy() - pred_test.detach().to('cpu').numpy())
acc = sum(validation_actual == validation_pred) / pred_test.shape[0]
acc_result.append(acc)
print(str(acc_result))