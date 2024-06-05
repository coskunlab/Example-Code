import torch
import torch_geometric
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import lightning.pytorch as pl
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn as nn
from torch_geometric.utils import  add_self_loops, softmax
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn import MessagePassing, GCNConv, GraphConv
# from torchmetrics.classification import BinaryF1Score, BinaryAUROC
from torchmetrics import AUROC, F1Score, Accuracy
import math 

### MLP 

class MLPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.n_feat = in_channels
        self.n_classes = out_channels
        self.n_hidden = hidden_channels
        self.n_layers = num_layers

        self.layers = [Linear(self.n_feat, self.n_hidden)]
        if self.n_layers > 2:
            for _ in range(self.n_layers-2):
                self.layers.append(Linear(self.n_hidden, self.n_hidden))
        self.layers.append(Linear(self.n_hidden, self.n_classes))
        self.layers = torch.nn.Sequential(*self.layers)
    
    def forward(self, x, *args, **kwargs):
        for layer in range(self.n_layers):
            x = self.layers[layer](x)
            if layer == self.n_layers - 1:
                #remove relu for the last layer
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)
        return x

class MLPModule(pl.LightningModule):
    def __init__(self, c_out=2, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.model = MLPModel(**model_kwargs)

    def forward(self, data, mode="train"):
        x = data.x
        out = self.model(x)

        loss = self.loss_module(out, data.y)
        pred = out.argmax(dim=1)
        acc = (pred == data.y).sum().float() / len(pred)
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)

### GCN 

gnn_layer_by_name = {"GCN": GCNConv, 'GraphConv': GraphConv}

class GNNModel(torch.nn.Module):
    def __init__(self, layer_name, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.n_feat = in_channels
        self.n_classes = out_channels
        self.n_hidden = hidden_channels
        self.n_layers = num_layers

        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = self.n_feat, self.n_hidden
        
        for _ in range(self.n_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(self.dropout),
                torch.nn.BatchNorm1d(out_channels)
            ]
            in_channels = self.n_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=self.n_classes , **kwargs)]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, torch_geometric.nn.MessagePassing):
                x = layer(x, edge_index, edge_weight=edge_weight)
            else:
                x = layer(x)
        return x

"""Attention pooling module"""
class Attention_module(Aggregation):
    def __init__(self, D1 = 20, D2 = 10):
        super(Attention_module, self).__init__()
        self.attention_Tanh = [
            nn.Linear(D1, D2),
            nn.Tanh()]
        
        self.attention_Sigmoid = [
            nn.Linear(D1, D2),
            nn.Sigmoid()]

        self.attention_Tanh = nn.Sequential(*self.attention_Tanh)
        self.attention_Sigmoid = nn.Sequential(*self.attention_Sigmoid)
        self.attention_Concatenate = nn.Linear(D2, 1)

    def forward(self, x, index=None, ptr=None, dim_size = None, dim= -2): # 20->10->2
        tanh_res = self.attention_Tanh(x)
        sigmoid_res = self.attention_Sigmoid(x)
        Attention_score = tanh_res.mul(sigmoid_res)
        Attention_score = self.attention_Concatenate(Attention_score)  # N x n_classes

        # return Attention_score, x
        gate = softmax(Attention_score, index, ptr, dim_size, dim)
        return self.reduce(gate * x, index, ptr, dim_size, dim)

"""Initial weights"""
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class GraphLevelGNN(pl.LightningModule):
    def __init__(self, model_name, num_PPI_type, c_out=2, graph_pooling="mean", embedding=True, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        c_hidden = model_kwargs.get('in_channels', 16) # Output dimension of GCN layers
        out_channels = model_kwargs.get('out_channels', 16)
        self.model_name = model_name

        if embedding:
            self.x_embedding = torch.nn.Linear(num_PPI_type, c_hidden)
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        else: 
            self.x_embedding = nn.Identity()

        if model_name == "MLP":
            print('Using MLP')
            self.model = MLPModel(**model_kwargs)
        elif model_name == 'GINConv':
            print('GINConv')
            self.model = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', edge_dim=2,**model_kwargs)
        elif model_name == 'GINConv_norm':
            print('GINConv')
            self.model = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', edge_dim=1,**model_kwargs)
        elif model_name == 'GAT' :
            print('Using GAT')
            self.model = torch_geometric.nn.models.GAT(dropout=0.5, norm='BatchNorm', v2=True, heads=8, edge_dim=2,**model_kwargs)
        elif model_name == 'GAT_norm':
            print('Using GAT')
            self.model = torch_geometric.nn.models.GAT(dropout=0.5, norm='BatchNorm', v2=True, heads=8, edge_dim=1,**model_kwargs)
        else:
            print('Using GNN')
            self.model = GNNModel(layer_name=model_name, **model_kwargs)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(out_channels, out_channels//2), 
                                        torch.nn.Dropout(0.5), torch.nn.Linear(out_channels//2, c_out))
        
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.valid_auroc = AUROC(task="binary")
        self.valid_f1 = F1Score(task="binary")

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(out_channels, 1))
        elif graph_pooling == "attention2":
            self.pool = Attention_module(D1 = out_channels, D2=out_channels//2)
        else:
            raise ValueError("Invalid graph pooling type.")


    def forward(self, data, mode="train"):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)

        x = self.x_embedding(x.float())
        if self.model_name == 'GINConv' or self.model_name == 'GAT':
            x = self.model(x , edge_index, edge_attr=edge_attr)
        elif self.model_name == 'GINConv_norm' or self.model_name == 'GAT_norm':
            x = self.model(x , edge_index, edge_attr=edge_weight)
        else:
            x = self.model(x , edge_index, edge_weight=edge_weight)
        x = self.pool(x, batch) 
        out = self.head(x)

        # pred = out.argmax(dim=1)
        # acc = (pred == data.y).sum().float() / len(pred)
        # f1 = self.metricf1(pred, data.y)
        # auc = self.metricauc(pred, data.y)
        return out

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # print('1 Step')
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)
        self.train_acc(y_hat, y)
        self.train_auroc(out[:,1], y)
        self.train_f1(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)
        self.valid_acc(y_hat, y)
        self.valid_auroc(out[:,1], y)
        self.valid_f1(y_hat, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.valid_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
class BilinearFusion(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=16, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 32), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(32+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out

class GraphLevelGNN2D3DFusion(pl.LightningModule):
    def __init__(self, model_name, num_PPI_type, c_out=2, graph_pooling="mean",  fusion='concat', **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        c_hidden = model_kwargs.get('in_channels', 16) # Output dimension of GCN layers
        out_channels = model_kwargs.get('out_channels', 16)
        self.model_name = model_name
        self.fusion = fusion
        self.x_embedding = nn.Identity()

        # Create 2 model 1 for 2D and 1 for 3D
        if model_name == "MLP":
            print('Using MLP')
            self.model2D = MLPModel(**model_kwargs)
            self.model3D = MLPModel(**model_kwargs)
        elif model_name == 'GINConv':
            print('GINConv')
            self.model2D = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', edge_dim=2,**model_kwargs)
            self.model3D = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', edge_dim=2,**model_kwargs)
        elif model_name == 'GINConv':
            print('GINConv')
            self.model2D = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', edge_dim=1,**model_kwargs)
            self.model3D = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', edge_dim=1,**model_kwargs)
        elif model_name == 'GAT':
            print('Using GAT')
            self.model2D = torch_geometric.nn.models.GAT(dropout=0.5, norm='BatchNorm', v2=True, heads=4, edge_dim=1,**model_kwargs)
            self.model3D = torch_geometric.nn.models.GAT(dropout=0.5, norm='BatchNorm', v2=True, heads=4, edge_dim=1,**model_kwargs)
        else:
            print('Using GNN')
            self.model2D = GNNModel(layer_name=model_name, **model_kwargs)
            self.model3D = GNNModel(layer_name=model_name, **model_kwargs)
        
        # Fusion head
        if self.fusion=='concat':
            self.head = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(out_channels*2, out_channels), 
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Dropout(0.5), torch.nn.Linear(out_channels, c_out))
        elif self.fusion == 'bilinear':
            self.head = BilinearFusion(dim1= out_channels, dim2= out_channels, 
                                       scale_dim1=1, gate1=1, 
                                       scale_dim2=1, gate2=1, 
                                       skip=True, mmhid= out_channels)
            self.pred = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(out_channels, c_out))
        
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.valid_auroc = AUROC(task="binary")
        self.valid_f1 = F1Score(task="binary")

        if graph_pooling == "sum":
            self.pool2D = global_add_pool
            self.pool3D = global_add_pool
        elif graph_pooling == "mean":
            self.pool2D = global_mean_pool
            self.pool3D = global_mean_pool
        elif graph_pooling == "max":
            self.pool2D = global_max_pool
            self.pool3D = global_max_pool
        elif graph_pooling == "attention":
            self.pool2D = GlobalAttention(gate_nn = torch.nn.Linear(out_channels, 1))
            self.pool3D = GlobalAttention(gate_nn = torch.nn.Linear(out_channels, 1))
        elif graph_pooling == "attention2":
            self.pool2D = Attention_module(D1 = out_channels, D2=out_channels//2)
            self.pool3D = Attention_module(D1 = out_channels, D2=out_channels//2)
        else:
            raise ValueError("Invalid graph pooling type.")


    def forward(self, data, mode="train"):
        x, edge_index= data.x, data.edge_index, 
        edge_weight_2D = data.edge_weight_2D 
        edge_weight_3D = data.edge_weight_3D 

        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)

        x = self.x_embedding(x.float())
        # Pass into 2D and 3D model
        if self.model_name == 'GINConv' or self.model_name == 'GAT':
            x_2D = self.model2D(x , edge_index, edge_attr=edge_weight_2D)
            x_3D = self.model3D(x , edge_index, edge_attr=edge_weight_3D)
        else:
            x_2D = self.model2D(x , edge_index, edge_weight=edge_weight_2D)
            x_3D = self.model3D(x , edge_index, edge_weight=edge_weight_3D)

        # Get pooling 2D and 3D
        x_2D = self.pool2D(x_2D, batch) 
        x_3D = self.pool3D(x_3D, batch) 

        # Fusion 
        if self.fusion == 'concat':
            x = torch.concat((x_2D, x_3D), dim=1)
            # print(x.shape)
            out = self.head(x)
            out= F.softmax(out, dim = 1)
        elif self.fusion == 'bilinear':
            x = self.head(x_2D, x_3D)
            out = self.pred(x)
            out= F.softmax(out, dim = 1)

        return out

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # print('1 Step')
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)
        self.train_acc(y_hat, y)
        self.train_auroc(out[:,1], y)
        self.train_f1(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)
        self.valid_acc(y_hat, y)
        self.valid_auroc(out[:,1], y)
        self.valid_f1(y_hat, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.valid_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    