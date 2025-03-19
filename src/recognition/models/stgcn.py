import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgcnutils.tgcn import ConvTemporalGraphical
from .stgcnutils.graph import Graph

__all__ = ["STGCN"]

MAX_NUM_CLASSES = 4

class STGCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, device, **kwargs):
        super().__init__()

        self.device = device
        self.num_class = num_class

        #num_class = 27 #todo temp mismatch between act recog model and VAE model
        
        self.losses = ["accuracy", "cross_entropy", "mixed"]
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9 #todo orig 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(1792, num_class * MAX_NUM_CLASSES, kernel_size=1) #todo length=60
        self.fcn = nn.Conv2d(2560, num_class * MAX_NUM_CLASSES, kernel_size=1) #length = 80

        #self.fcn = nn.Conv2d(3840, num_class * MAX_NUM_CLASSES, kernel_size=1) #length=120
        #self.fcn = nn.Conv2d(5120, num_class * MAX_NUM_CLASSES, kernel_size=1) #length=160
        
        #self.fcn = nn.Conv2d(6400, num_class * MAX_NUM_CLASSES, kernel_size=1) #length 200, 4 classes
        #self.fcn = nn.Conv2d(12800, num_class * MAX_NUM_CLASSES, kernel_size=1) #8 ACTS length=160: 5120, 1 ACT length 160: 5120

        
        #self.fcn = nn.Conv2d(5120, num_class * MAX_NUM_CLASSES, kernel_size=1) #8 ACTS length=160: 5120, 1 ACT length 160: 5120

        


    def forward(self, batch):
        # TODO: use mask
        # Received batch["x"] as
        #   Batch(48), Joints(23), Quat(4), Time(157
        # Expecting:
        #   Batch, Quat:4, Time, Joints, 1
        x = batch["x"].permute(0, 2, 3, 1).unsqueeze(4).contiguous()

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        if x.shape[2] == 23:
            x = x[:,0:22]
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # compute feature
        # _, c, t, v = x.size()
        # features = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        # batch["features"] = features
        
        # global pooling
        #x = F.avg_pool2d(x, x.size()[2:]) #todo
        x = F.avg_pool2d(x, [2,x.size()[3]])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # features
        batch["features"] = x.squeeze()
        
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        batch["yhat"] = x
        return batch

    def compute_accuracy(self, batch):
        total_acc = 0
        for c in range(MAX_NUM_CLASSES):
            confusion = torch.zeros(self.num_class, self.num_class, dtype=int)
            if True or MAX_NUM_CLASSES > 1: #todo
                yhat = batch["yhat"][:,c*self.num_class: (c+1)*self.num_class].max(dim=1).indices
                ygt = batch["y"][:,c]
            else:
                yhat = batch["yhat"].max(dim=1).indices
                ygt = batch["y"]
            for label, pred in zip(ygt, yhat):
                confusion[label][pred] += 1
            accuracy = torch.trace(confusion)/torch.sum(confusion)
            total_acc += accuracy
        return total_acc/MAX_NUM_CLASSES
    
    def compute_loss(self, batch):
        if False and MAX_NUM_CLASSES == 1:
            cross_entropy = self.criterion(batch["yhat"], batch["y"])
        else:
            num_classes = batch['y'].shape[1]
            cross_entropy = 0
            for c in range(num_classes):
                current_yhat = batch["yhat"][:,c*self.num_class: (c+1)*self.num_class]
                cross_entropy += self.criterion(current_yhat, batch["y"][:,c])

        mixed_loss = cross_entropy
        
        acc = self.compute_accuracy(batch)
        losses = {"cross_entropy": cross_entropy.item(),
                  "mixed": mixed_loss.item(),
                  "accuracy": acc.item()}
        return mixed_loss, losses


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


if __name__ == "__main__":
    model = STGCN(in_channels=3, num_class=60, edge_importance_weighting=True, graph_args={"layout": "smpl_noglobal", "strategy": "spatial"})
    # Batch, in_channels, time, vertices, M
    inp = torch.rand(10, 3, 16, 23, 1)
    out = model(inp)
    print(out.shape)
    import pdb
    pdb.set_trace()



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .stgcnutils.tgcn import ConvTemporalGraphical
# from .stgcnutils.graph import Graph
# import numpy as np
# __all__ = ["STGCN"]

# MAX_NUM_CLASSES = 8
# NUM_FRAMES = 120
# class STGCN(nn.Module):
#     r"""Spatial temporal graph convolutional networks.
#     Args:
#         in_channels (int): Number of channels in the input data
#         num_class (int): Number of classes for the classification task
#         graph_args (dict): The arguments for building the graph
#         edge_importance_weighting (bool): If ``True``, adds a learnable
#             importance weighting to the edges of the graph
#         **kwargs (optional): Other parameters for graph convolution units
#     Shape:
#         - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
#         - Output: :math:`(N, num_class)` where
#             :math:`N` is a batch size,
#             :math:`T_{in}` is a length of input sequence,
#             :math:`V_{in}` is the number of graph nodes,
#             :math:`M_{in}` is the number of instance in a frame.
#     """

#     def __init__(self, in_channels, num_class, graph_args,
#                  edge_importance_weighting, device, **kwargs):
#         super().__init__()

#         self.device = device
#         self.num_class = num_class
        
#         self.losses = ["accuracy", "cross_entropy", "mixed"]
#         self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

#         # load graph
#         self.graph = Graph(**graph_args)
#         A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
#         self.register_buffer('A', A)

#         # build networks
#         spatial_kernel_size = A.size(0)
#         temporal_kernel_size = 9 #todo orig 9
#         kernel_size = (temporal_kernel_size, spatial_kernel_size)
#         self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
#         kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
#         self.st_gcn_networks = nn.ModuleList((
#             st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
#             st_gcn(64, 64, kernel_size, 1, **kwargs),
#             st_gcn(64, 64, kernel_size, 1, **kwargs),
#             st_gcn(64, 64, kernel_size, 1, **kwargs),
#             st_gcn(64, 128, kernel_size, 1, **kwargs), #todo stride 2->1 to keep the temporal dimension
#             st_gcn(128, 128, kernel_size, 1, **kwargs),
#             st_gcn(128, 128, kernel_size, 1, **kwargs),
#             st_gcn(128, 256, kernel_size, 1, **kwargs), #todo stride 2->1 to keep the temporal dimension
#             st_gcn(256, 256, kernel_size, 1, **kwargs),
#             st_gcn(256, 256, kernel_size, 1, **kwargs),
#         ))

#         # initialize parameters for edge importance weighting
#         if edge_importance_weighting:
#             self.edge_importance = nn.ParameterList([
#                 nn.Parameter(torch.ones(self.A.size()))
#                 for i in self.st_gcn_networks
#             ])
#         else:
#             self.edge_importance = [1] * len(self.st_gcn_networks)

#         # fcn for prediction
#         self.fcn = nn.Conv2d(256, num_class*MAX_NUM_CLASSES, kernel_size=1)

#     def forward(self, batch):
#         # TODO: use mask
#         # Received batch["x"] as
#         #   Batch(48), Joints(23), Quat(4), Time(157
#         # Expecting:
#         #   Batch, Quat:4, Time, Joints, 1
#         x = batch["x"].permute(0, 2, 3, 1).unsqueeze(4).contiguous()

#         # data normalization
#         N, C, T, V, M = x.size()
#         x = x.permute(0, 4, 3, 1, 2).contiguous()
#         x = x.view(N * M, V * C, T)
#         if x.shape[2] == 23:
#             x = x[:,0:22]
#         x = self.data_bn(x)
#         x = x.view(N, M, V, C, T)
#         x = x.permute(0, 1, 3, 4, 2).contiguous()
#         x = x.view(N * M, C, T, V)

#         # forward
#         for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
#             x, _ = gcn(x, self.A * importance)

#         # compute feature
#         # _, c, t, v = x.size()
#         # features = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
#         # batch["features"] = features
        
#         # global pooling
#         x = F.avg_pool2d(x, x.size()[2:])
#         #x = F.avg_pool2d(x, (1,x.size()[3])) #todo
#         x = x.view(N, M, -1, 1, 1).mean(dim=1)

#         # features
#         batch["features"] = x.squeeze()
        
#         # prediction
#         x = self.fcn(x)
#         x = x.view(x.size(0), -1)
#         batch["yhat"] = x.reshape(x.shape[0], MAX_NUM_CLASSES*self.num_class,-1,1)
#         return batch

#     def compute_accuracy(self, batch):
#         total_acc = 0
#         for c in range(MAX_NUM_CLASSES):
#             confusion = torch.zeros(self.num_class, self.num_class, dtype=int)
#             if True or MAX_NUM_CLASSES > 1: #todo
#                 yhat = batch["yhat"][:,c*self.num_class: (c+1)*self.num_class].max(dim=1).indices
#                 ygt = batch["y"][:,c]
#             else:
#                 yhat = batch["yhat"].max(dim=1).indices
#                 ygt = batch["y"]
#             for label, pred in zip(ygt, yhat):
#                 confusion[label][pred] += 1
#             accuracy = torch.trace(confusion)/torch.sum(confusion)
#             total_acc += accuracy
#         return accuracy/MAX_NUM_CLASSES
    
#     def compute_loss_weakly(self, batch): #using the paper for multiple action sequences 
#         #https://openaccess.thecvf.com/content_WACV_2020/papers/Miki_Weakly_Supervised_Graph_Convolutional_Neural_Network_for_Human_Action_Localization_WACV_2020_paper.pdf
#         y_hat = batch['yhat']
#         y = batch['y']
#         loss_cum = 0
#         mu1 = 1e-5
#         mu2 = 1e-2
#         mu3 = 1e-2

#         log_term = 0
#         all_pos_ind = []
#         all_neg_ind = []
#         all_acts_exist = torch.zeros(y.shape[0]).to(y.device).to(torch.bool)
#         #for n in range(y_hat.shape[2]):
#             # for k in range(y.shape[1]):
#             #     act_exists = y[:,k]==n
#             #     all_acts_exist = all_acts_exist.bitwise_or(act_exists)
#             # all_acts_exist = all_acts_exist[:,].type(torch.int)  

#             # pos_ind = torch.where(all_acts_exist==1)[0]
#             # neg_ind = torch.where(all_acts_exist==0)[0]
            
#             # if len(pos_ind) == 0 or len(neg_ind) == 0:
#             #     all_pos_ind.append(-1)
#             #     all_neg_ind.append(-1)
#             #     continue
#             # rnd_p_index = np.random.choice(pos_ind.cpu())
#             # rnd_n_index = np.random.choice(neg_ind.cpu())
            
#             # all_pos_ind.append(rnd_p_index)
#             # all_neg_ind.append(rnd_n_index)

#             # y_n_max = y_hat[rnd_p_index,:,n].max(dim=0)
#             # y_n_maxv = y_n_max[0][0].float()
#             # log_term += torch.exp(y_n_maxv)

#             # y_n_max = y_hat[rnd_n_index,:,n].max(dim=0)
#             # y_n_maxv = y_n_max[0][0].float()
#             # log_term += torch.exp(y_n_maxv)


#         for n in range(y_hat.shape[2]):
#             y_n_max = y_hat[:,:,n].max(dim=1)
#             y_n_maxv = y_n_max[0][:, 0].float()
#             all_acts_exist = torch.zeros(y.shape[0]).to(y.device).to(torch.bool)
#             for k in range(y.shape[1]):
#                 act_exists = y[:,k]==n
#                 all_acts_exist = all_acts_exist.bitwise_or(act_exists)
#             all_acts_exist = all_acts_exist[:,].type(torch.int)  
#             # all_acts_exist1 = all_acts_exist.clone()
#             # all_acts_exist1[all_acts_exist1==0] = -1

#             pos_ind = torch.where(all_acts_exist==1)[0]
#             neg_ind = torch.where(all_acts_exist==0)[0]
#             if len(pos_ind) == 0 or len(neg_ind) == 0:
#                continue
#             rnd_p_index = np.random.choice(pos_ind.cpu())
#             rnd_n_index = np.random.choice(neg_ind.cpu())

#             #rnd_p_index = all_pos_ind[n]
#             #rnd_n_index = all_neg_ind[n]
#             if rnd_p_index <0 or rnd_n_index < 0:
#                 continue
#             lamb1 = 0
#             lamb2 = 0
#             lamb3 = 0
#             current_loss = 0
#             for k in [rnd_p_index, rnd_n_index]:
#                 #lamb1 +=  torch.stack([y_hat[k,t,n]- y_hat[k,t+1,n] for t in range(y_hat.shape[1]-1)]).sum()*mu1
#                 for t in range(y_hat.shape[1]-1):
#                     lamb1 +=  (y_hat[k,t,n]- y_hat[k,t+1,n]).sum()
#                     lamb2 +=  y_hat[k,t,n].sum()
#                 phi = torch.tensor((k==0)).type(torch.int)
#                 lamb2 += y_hat[k,-1,n].sum()
#                 lamb1 *= mu1
#                 lamb2 *= mu2
#                 #lamb3 +=  (phi * torch.log(torch.exp(y_n_maxv)/log_term)).sum() *mu3
                
#                 phi1 = phi if phi == 1 else -phi
#                 current_loss += max(torch.tensor(0).to(y_hat.device).float(), (phi - phi1*y_n_maxv[k]))# + lamb1 +lamb2 #+ lamb2#-lamb3
#             loss_cum += current_loss

#         #loss_cum /= y.shape[0]
#         acc = self.compute_accuracy(batch)
#         print('loss_cum',loss_cum.item(), ', acc: ', acc)

#         losses = {"cross_entropy": loss_cum.item(),
#                   "mixed": loss_cum.item(),
#                   "accuracy": acc.item()}
#         return loss_cum, losses


#     def compute_loss(self, batch):
#         if False and MAX_NUM_CLASSES == 1:
#             cross_entropy = self.criterion(batch["yhat"], batch["y"])
#         else:
#             cross_entropy = self.criterion(batch["yhat"].reshape(batch["yhat"].shape[0], -1, MAX_NUM_CLASSES), batch["y"])
#         mixed_loss = cross_entropy
        
#         acc = self.compute_accuracy(batch)
#         losses = {"cross_entropy": cross_entropy.item(),
#                   "mixed": mixed_loss.item(),
#                   "accuracy": acc.item()}
#         return mixed_loss, losses



# class st_gcn(nn.Module):
#     r"""Applies a spatial temporal graph convolution over an input graph sequence.
#     Args:
#         in_channels (int): Number of channels in the input sequence data
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
#         stride (int, optional): Stride of the temporal convolution. Default: 1
#         dropout (int, optional): Dropout rate of the final output. Default: 0
#         residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
#     Shape:
#         - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
#         - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
#         - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
#         - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
#         where
#             :math:`N` is a batch size,
#             :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
#             :math:`T_{in}/T_{out}` is a length of input/output sequence,
#             :math:`V` is the number of graph nodes.
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  dropout=0,
#                  residual=True):
#         super().__init__()

#         assert len(kernel_size) == 2
#         assert kernel_size[0] % 2 == 1
#         padding = ((kernel_size[0] - 1) // 2, 0)

#         self.gcn = ConvTemporalGraphical(in_channels, out_channels,
#                                          kernel_size[1])

#         self.tcn = nn.Sequential(
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 out_channels,
#                 out_channels,
#                 (kernel_size[0], 1),
#                 (stride, 1),
#                 padding,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.Dropout(dropout, inplace=True),
#         )

#         if not residual:
#             self.residual = lambda x: 0

#         elif (in_channels == out_channels) and (stride == 1):
#             self.residual = lambda x: x

#         else:
#             self.residual = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels,
#                     out_channels,
#                     kernel_size=1,
#                     stride=(stride, 1)),
#                 nn.BatchNorm2d(out_channels),
#             )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, A):

#         res = self.residual(x)
#         x, A = self.gcn(x, A)
#         x = self.tcn(x) + res

#         return self.relu(x), A


# if __name__ == "__main__":
#     model = STGCN(in_channels=3, num_class=60, edge_importance_weighting=True, graph_args={"layout": "smpl_noglobal", "strategy": "spatial"})
#     # Batch, in_channels, time, vertices, M
#     inp = torch.rand(10, 3, 16, 23, 1)
#     out = model(inp)
#     print(out.shape)
#     import pdb
#     pdb.set_trace()
