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
# Best Params from Optuna
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
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            checkpoint = torch.load(cost_model_path)
        if self.device.type == 'cpu':
            checkpoint = torch.load(cost_model_path,map_location=torch.device(self.device))
        
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

    def estimate(self, c_files, tir_files, sw_feats_list,run_ids):
        """
        Estimates runtime and code size for a batch of c_files and tir_files.

        Parameters:
        - c_files (list): A list of paths to the C code files.
        - tir_files (list): A list of paths to the TIR (TVM IR) files.
        - sw_feats_list (list): A list of software feature vectors (e.g., [0, 1, 0, 1]),
                                one for each file pair.
        - run_ids (list): List of valid run_idx

        Returns:
        - list: A list of (predicted_runtime, predicted_code_size) tuples.
                If a graph fails to be created, its position in the list
                will contain None.
        """
        if not (len(c_files) == len(tir_files) == len(sw_feats_list)):
            raise ValueError("Input lists (c_files, tir_files, sw_feats_list) must have the same length.")

        graph_list = []
        global_feats_list = []
        valid_indices = []  # To track which graphs were successfully created

        
        for i, (c_file, tir_file, sw_feats,run_id) in enumerate(zip(c_files, tir_files, sw_feats_list,run_ids)):
            try:
                mods = load_tir(tir_file)
                if not mods:
                    logging.warning(f"No TIR mods loaded from {tir_file}, skipping index {i}")
                    continue
                
                func_name_to_feat = tir_tofeats_mlp(mods, self.filter_rows)
                nodes, edges = parse_graph_from_c_code(c_file, func_name_to_feat)
                
                if not nodes: # Check if graph parsing failed
                    logging.warning(f"Failed to parse graph from {c_file}, skipping index {i}")
                    continue
                
                g = create_dgl_graph(nodes, edges)
                
                
                if hasattr(g, 'ndata') and 'feat' in g.ndata:
                    feats_np = g.ndata['feat'].numpy()
                    scaled_feats = self.feature_scaler.transform(feats_np)
                    g.ndata['feat'] = torch.tensor(scaled_feats, dtype=torch.float32)
                
                if hasattr(g, 'edata') and 'weight' in g.edata:
                    ew_np = g.edata['weight'].numpy().reshape(-1, 1)
                    scaled_ew = self.edge_scaler.transform(ew_np).flatten()
                    g.edata['weight'] = torch.tensor(scaled_ew, dtype=torch.float32)

                
                global_sw_flags_tensor = torch.tensor(np.array(sw_feats), dtype=torch.float32).unsqueeze(0)
                
                
                graph_list.append(g)
                global_feats_list.append(global_sw_flags_tensor)
                valid_indices.append(run_id)
            
            except Exception as e:
                logging.error(f"Error processing files {c_file}/{tir_file} at index {i}: {e}", exc_info=True)
                continue

        if not graph_list:
            logging.warning("No valid graphs were created for estimation.")
            return [None] * len(c_files)  # Return list of Nones matching input length

        
        self.model.eval()  # Set model to evaluation mode
        results = []
        final_results = {}

        with torch.no_grad():
            # Batch graphs and global features
            batched_g = dgl.batch(graph_list).to(self.device)
            batched_global_feats = torch.cat(global_feats_list, dim=0).to(self.device)

            # Run batch inference
            output = self.model(batched_g, batched_global_feats)
            output_np = output.cpu().numpy()
            
            
            inv_transformed = self.target_scaler.inverse_transform(output_np)
            
            # De-logarithmize (element-wise)
            predicted_runtimes = 10 ** inv_transformed[:, 0]
            predicted_code_sizes = 10 ** inv_transformed[:, 1]
            
            # Zip results into (runtime, codesize) tuples
            results = list(zip(predicted_runtimes, predicted_code_sizes))
            
            
            for i, res in zip(valid_indices, results):
                final_results[i] = res

        return final_results # Dict of 