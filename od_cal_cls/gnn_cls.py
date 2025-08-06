"""
IMPORT MODULES
"""

import os
import os.path as osp
from turtle import forward
import torch
import torch.nn.functional as nnF
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, Linear
from torch_geometric.data import Dataset, InMemoryDataset


"""
DEFINE DATASET & DATALOADER CLASSES
"""

# Class for importing Graphs.
class od_flow_graphs(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        lst_graphs = os.listdir(self.processed_dir)
        lst_graphs_fil = [i for i in lst_graphs if "graph" in i]        
        return lst_graphs_fil

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     path = download_url(url, self.raw_dir)
    #     ...

    # def process(self):
    #     idx = 0
    #     for raw_path in self.raw_paths:
    #         # Read data from `raw_path`.
    #         data = Data(...)

    #         if self.pre_filter is not None and not self.pre_filter(data):
    #             continue

    #         if self.pre_transform is not None:
    #             data = self.pre_transform(data)

    #         torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
    #         idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'graph_{:04d}.pt'.format(idx+1)))
        return data

# Class for importing Graphs, In Memory.
class od_flow_graphs_inMemory(InMemoryDataset):
    
    def __init__(self, root:str, data_list:list, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_list = data_list
        self.data, self.slices = torch.load(self.processed_paths[0])
    
# Class for GCN model.
class gnn_GCN_CONV_LIN(torch.nn.Module):
    
    def __init__(self,  in_dim_x: int, in_dim_y: int, in_dim_hid: int, 
                        in_num_layers: int = 2, in_lc_norm: bool = False, in_lc_dropout: bool = False,
                        in_rat_dropout: float = 0.0, 
        ) -> None:
        
        super().__init__()
        # Here, node level layers should be defined.
        # Number of GCN layers & Normarlization layers can be adjusted.

        # Graph Convolution layers.
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_dim_x, in_dim_hid)] + 
            [GCNConv(in_dim_hid, in_dim_hid) for i in range(in_num_layers - 2)] +
            [GCNConv(in_dim_hid, in_dim_hid)]
        )

        # Graph Linear layer as final one.
        self.out_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_y)
        
        # Node feature normalization layers. 
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features= in_dim_hid) for i in range(in_num_layers - 1)]
        )
        
        # Store internal attributes.
        self.lv_norm = in_lc_norm
        self.lv_dropout = in_lc_dropout        
        self.rat_dropout = in_rat_dropout
        
    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        self.out_linear.reset_parameters()
    
    def forward(self, loaded_data):
        # Load node feature and edge index information.
        node_x = loaded_data.x.type(torch.float)
        edge_idx = loaded_data.edge_index
        # Stack layers.
        if self.lv_norm and self.lv_dropout:            # Use normalisation & drop-out.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                if self.training:
                    node_x_trans = nnF.dropout(node_x_trans, p= self.rat_dropout)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        elif self.lv_norm and not(self.lv_dropout):     # Use normalisation.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        else:                                           # Just basics...
            for conv in self.convs[:-1]:
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        # Return output.
        return out

# Class to observe how GCN layer is working.
class gnn_GCN_CONV_test(torch.nn.Module):
    
    def __init__(self,  in_dim_x: int, in_dim_y: int, in_dim_hid: int, 
                        in_num_layers: int = 2, in_lc_norm: bool = False, in_lc_dropout: bool = False,
                        in_rat_dropout: float = 0.0, 
        ) -> None:
        
        super().__init__()
        # Here, node level layers should be defined.
        # Number of GCN layers & Normarlization layers can be adjusted.

        # Graph Convolution layers.
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_dim_x, in_dim_hid)] + 
            [GCNConv(in_dim_hid, in_dim_hid) for i in range(in_num_layers - 2)] +
            [GCNConv(in_dim_hid, in_dim_hid)]
        )

        # Graph Linear layer as final one.
        self.out_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_y)
        
        # Node feature normalization layers. 
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features= in_dim_hid) for i in range(in_num_layers - 1)]
        )
        
        # Store internal attributes.
        self.lv_norm = in_lc_norm
        self.lv_dropout = in_lc_dropout        
        self.rat_dropout = in_rat_dropout
        
    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        self.out_linear.reset_parameters()
    
    def forward(self, loaded_data):
        # Load node feature and edge index information.
        node_x = loaded_data.x.type(torch.float)
        edge_idx = loaded_data.edge_index
        # Stack layers.
        conv_1 = self.convs[0]
        out = conv_1(node_x, edge_idx)
        
        return out

# Class for GAT model.
class gnn_GAT_CONV_LIN(torch.nn.Module):
    
    def __init__(self,  in_dim_x: int, in_dim_y: int, in_dim_hid: int, in_neg_slope: float = 0.0,
                        in_num_layers: int = 2, in_lc_norm: bool = False, in_lc_dropout: bool = False,
                        in_rat_dropout: float = 0.0, 
        ) -> None:
        
        super().__init__()
        # Here, node level layers should be defined.
        # Number of GCN layers & Normarlization layers can be adjusted.

        # Graph Convolution layers.
        # For OD & Flow regression model, no negatie slope applied.
        self.convs = torch.nn.ModuleList(
            [GATConv(in_dim_x, in_dim_hid, negative_slope= in_neg_slope)] + 
            [GATConv(in_dim_hid, in_dim_hid, negative_slope= in_neg_slope) for i in range(in_num_layers - 2)] +
            [GATConv(in_dim_hid, in_dim_hid, negative_slope= in_neg_slope)]
        )

        # Graph Linear layer as final one.
        self.out_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_y)
        
        # Node feature normalization layers. 
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features= in_dim_hid) for i in range(in_num_layers - 1)]
        )
        
        # Store internal attributes.
        self.lv_norm = in_lc_norm
        self.lv_dropout = in_lc_dropout        
        self.rat_dropout = in_rat_dropout
        
    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        self.out_linear.reset_parameters()
    
    def forward(self, loaded_data):
        # Load node feature and edge index information.
        node_x = loaded_data.x.type(torch.float)
        edge_idx = loaded_data.edge_index
        # Stack layers.
        if self.lv_norm and self.lv_dropout:            # Use normalisation & drop-out.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                if self.training:
                    node_x_trans = nnF.dropout(node_x_trans, p= self.rat_dropout)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        elif self.lv_norm and not(self.lv_dropout):     # Use normalisation.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        else:                                           # Just basics...
            for conv in self.convs[:-1]:
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        # Return output.
        return out

# Class for GATv2.
class gnn_GATv2_CONV_LIN(torch.nn.Module):
    
    def __init__(self,  in_dim_x: int, in_dim_y: int, in_dim_hid: int, in_neg_slope: float = 0.0,
                        in_num_layers: int = 2, in_lc_norm: bool = False, in_lc_dropout: bool = False,
                        in_rat_dropout: float = 0.0, 
        ) -> None:
        
        super().__init__()
        # Here, node level layers should be defined.
        # Number of GCN layers & Normarlization layers can be adjusted.

        # Graph Convolution layers.
        # For OD & Flow regression model, no negatie slope applied.
        self.convs = torch.nn.ModuleList(
            [GATv2Conv(in_dim_x, in_dim_hid, negative_slope= in_neg_slope)] + 
            [GATv2Conv(in_dim_hid, in_dim_hid, negative_slope= in_neg_slope) for i in range(in_num_layers - 2)] +
            [GATv2Conv(in_dim_hid, in_dim_hid, negative_slope= in_neg_slope)]
        )

        # Graph Linear layer as final one.
        self.out_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_y)
        
        # Node feature normalization layers. 
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features= in_dim_hid) for i in range(in_num_layers - 1)]
        )
        
        # Store internal attributes.
        self.lv_norm = in_lc_norm
        self.lv_dropout = in_lc_dropout        
        self.rat_dropout = in_rat_dropout
        
    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        self.out_linear.reset_parameters()
    
    def forward(self, loaded_data):
        # Load node feature and edge index information.
        node_x = loaded_data.x.type(torch.float)
        edge_idx = loaded_data.edge_index
        # Stack layers.
        if self.lv_norm and self.lv_dropout:            # Use normalisation & drop-out.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                if self.training:
                    node_x_trans = nnF.dropout(node_x_trans, p= self.rat_dropout)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        elif self.lv_norm and not(self.lv_dropout):     # Use normalisation.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        else:                                           # Just basics...
            for conv in self.convs[:-1]:
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        # Return output.
        return out

# Class for GCN with 2 LIN layer.
class gnn_GCN_CONV_LIN2(torch.nn.Module):
    
    def __init__(self,  in_dim_x: int, in_dim_y: int, in_dim_hid: int, 
                        in_num_layers: int = 2, in_lc_norm: bool = False, in_lc_dropout: bool = False,
                        in_rat_dropout: float = 0.0, 
        ) -> None:
        
        super().__init__()
        # Here, node level layers should be defined.
        # Number of GCN layers & Normarlization layers can be adjusted.

        # Graph Convolution layers.
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_dim_x, in_dim_hid)] + 
            [GCNConv(in_dim_hid, in_dim_hid) for i in range(in_num_layers - 2)] +
            [GCNConv(in_dim_hid, in_dim_hid)]
        )

        # Graph Linear layer as final one.
        self.inter_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_hid)
        self.out_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_y)
        
        # Node feature normalization layers. 
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features= in_dim_hid) for i in range(in_num_layers - 1)]
        )
        
        # Store internal attributes.
        self.lv_norm = in_lc_norm
        self.lv_dropout = in_lc_dropout        
        self.rat_dropout = in_rat_dropout
        
    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        self.out_linear.reset_parameters()
    
    def forward(self, loaded_data):
        # Load node feature and edge index information.
        node_x = loaded_data.x.type(torch.float)
        edge_idx = loaded_data.edge_index
        # Stack layers.
        if self.lv_norm and self.lv_dropout:            # Use normalisation & drop-out.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                if self.training:
                    node_x_trans = nnF.dropout(node_x_trans, p= self.rat_dropout)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            node_x = self.inter_linear(node_x)
            out = self.out_linear(node_x)
        elif self.lv_norm and not(self.lv_dropout):     # Use normalisation.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            node_x = self.inter_linear(node_x)
            out = self.out_linear(node_x)
        else:                                           # Just basics...
            for conv in self.convs[:-1]:
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans
            node_x = self.convs[-1](node_x, edge_idx)
            node_x = self.inter_linear(node_x)
            out = self.out_linear(node_x)
        # Return output.
        return out
    
