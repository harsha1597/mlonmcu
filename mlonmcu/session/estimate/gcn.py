import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.nn.pytorch as dglnn
from dgl.dataloading import GraphDataLoader
import optuna
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from mlonmcu.session.estimate.extract_graph import load_tir,parse_graph_from_c_code, create_dgl_graph, tir_tofeats_mlp
from dgl.data import DGLDataset
import logging

logger = logging.getLogger(__name__)
# {'gcn_layers': 3,
#  'mlp_layers': 2,
#  'gcn_hidden_feats': 128,
#  'mlp_hidden_feats': 1024,
#  'lr': 0.0001757952049329151,
#  'weight_decay': 5.355392787707086e-06}

class GraphRegressionModel(nn.Module):

    def __init__(self, in_feats,  global_feats_dim,gcn_hidden_feats=128, gcn_layers=3, 
                 mlp_hidden_feats=1024, mlp_layers=2):
        """
        in_feats: Dimension of input node features
        gcn_hidden_feats: Dimension of hidden GCN layers
        gcn_layers: Number of GCN layers
        mlp_hidden_feats: Dimension of hidden MLP layers
        mlp_layers: Number of hidden layers in the prediction MLP
        global_feats_dim: Dimension of the global features to concatenate
        """
        super(GraphRegressionModel, self).__init__()
        
        # --- 1. Build GCN Layers ---
        self.conv_layers = nn.ModuleList()
        # Input layer
        self.conv_layers.append(dglnn.GraphConv(in_feats, gcn_hidden_feats, 
                                                allow_zero_in_degree=False))
        
        # Hidden GCN layers
        gcn_hidden_feats_prev= gcn_hidden_feats
        for _ in range(gcn_layers - 1):
            gcn_hidden_feats_next = gcn_hidden_feats # gcn_hidden_feats_prev // 2 if gcn_hidden_feats_prev > 16 else gcn_hidden_feats_prev
            self.conv_layers.append(dglnn.GraphConv(gcn_hidden_feats_prev, gcn_hidden_feats_next, 
                                                    allow_zero_in_degree=False))
            gcn_hidden_feats_prev = gcn_hidden_feats_next

        # --- 2. Build Prediction Head (MLP) ---
        self.prediction_head = nn.ModuleList()
        
        # Input layer for MLP
        mlp_input_dim = gcn_hidden_feats_next + global_feats_dim
        self.prediction_head.append(nn.Linear(mlp_input_dim, mlp_hidden_feats))
        self.prediction_head.append(nn.ReLU())
        self.prediction_head.append(nn.Dropout(0.3)) # Added Dropout for regularization

        # Hidden layers for MLP
        for _ in range(mlp_layers - 1): # -1 because we already have one input layer
            self.prediction_head.append(nn.Linear(mlp_hidden_feats, mlp_hidden_feats))
            self.prediction_head.append(nn.ReLU())
            self.prediction_head.append(nn.Dropout(0.3))

        # Output layer for MLP
        self.prediction_head.append(nn.Linear(mlp_hidden_feats, 2)) # Predict 2 target values
        
        # Combine MLP layers into a Sequential module for easy forward pass
        self.mlp_seq = nn.Sequential(*self.prediction_head)

    def forward(self, g, global_feats):
        """
        g: The batched graph
        global_feats: Batched global features [batch_size, global_feats_dim]
        """
        # Node features
        h = g.ndata['feat'].float() 

        # GCN message passing
        for conv in self.conv_layers:
            h = F.relu(conv(g, h)) 

        # Graph-level embedding by summing node features
        g.ndata['h'] = h
        graph_embedding = dgl.sum_nodes(g, 'h')
        
        # Concatenate graph embedding with global features
        combined_embedding = torch.cat([graph_embedding, global_feats], dim=1)
        
        # Pass through the prediction head
        return self.mlp_seq(combined_embedding)
  
class RuntimeCodeSizedataset(DGLDataset):
    def __init__(self, graph_list, labels, global_flags):
        self.graphs = graph_list
        self.labels = labels
        self.global_flags = global_flags
        super().__init__(name='runtime_prediction')

    def __getitem__(self, i):
        return self.graphs[i], self.global_flags[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

class EstimatePostBuild():
    def __init__(self, cost_model_path):
        
        # 1. Load the checkpoint dictionary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(cost_model_path,map_location=torch.device(self.device))
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.model = GraphRegressionModel(
            in_feats=130, 
            global_feats_dim=4
        )
        
        # 3. Load the model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        # 4. Load the other components
        self.target_scaler = checkpoint['target_scaler']
        self.feature_scaler = checkpoint['feature_scaler']
        self.edge_scaler = checkpoint['edge_scaler']
        self.filter_rows = checkpoint['filter_rows']
        
        # 5. Set model to evaluation mode
        self.model.eval()
        
# Example of using the class:
# estimator = EstimatePostBuild(cost_model_path="path/to/your/model_checkpoint.pt")
    def estimate(self,c_file,tir_file,sw_feats):
        mods = load_tir(tir_file)
        if not mods:
            logging.warning(f"No TIR mods loaded from {tir_file}")
            return None
        func_name_to_feat = tir_tofeats_mlp(mods, self.filter_rows)
        nodes, edges = parse_graph_from_c_code(c_file, func_name_to_feat)
        g = create_dgl_graph(nodes, edges)
        
        # Scale node features
        if hasattr(g, 'ndata') and 'feat' in g.ndata:
            feats_np = g.ndata['feat'].numpy()
            scaled_feats = self.feature_scaler.transform(feats_np)
            g.ndata['feat'] = torch.tensor(scaled_feats, dtype=torch.float32)
        
        # Scale edge weights if they exist
        if hasattr(g, 'edata') and 'weight' in g.edata:
            ew_np = g.edata['weight'].numpy().reshape(-1, 1)
            scaled_ew = self.edge_scaler.transform(ew_np).flatten()
            g.edata['weight'] = torch.tensor(scaled_ew, dtype=torch.float32)
        
        global_sw_flags_tensor = torch.tensor(np.array(sw_feats), dtype=torch.float32)
        self.model.eval() # Set model to evaluation mode
        
        with torch.no_grad():
            g = g.to(self.device)
            global_sw_flags_tensor = global_sw_flags_tensor.to(self.device).unsqueeze(0)  # Add batch dimension
            output = self.model(g, global_sw_flags_tensor)
            output_np = output.cpu().numpy()
            # Inverse transform the target
            inv_transformed = self.target_scaler.inverse_transform(output_np)
            predicted_runtime = 10 ** inv_transformed[0][0]  # Assuming the first output is runtime
            predicted_code_size = 10 ** inv_transformed[0][1]  # Assuming the second output is code size
        return predicted_runtime, predicted_code_size

