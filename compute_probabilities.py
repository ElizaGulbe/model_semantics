import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
import os
import glob

# Define the same model structure
def deserialize_tensor(serialized_tensor):
    return pickle.loads(bytes.fromhex(serialized_tensor))

class DynamicSemanticRelationModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes, activation, dropout_rate):
        super(DynamicSemanticRelationModel, self).__init__()
        
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(p=dropout_rate))  # Add dropout after each activation layer
            in_size = hidden_size
            
        layers.append(nn.Linear(in_size, num_classes))
        
        self.model = nn.Sequential(*layers)
        # Add Softmax layer
        self.softmax = nn.Softmax(dim=1)  # Apply softmax to the last dimension
    
    def forward(self, x):
        x = self.model(x)
        return self.softmax(x) 

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

experiment_path = "C:/ray_results/train_model_2024-08-27_00-40-32/rund238e_00000"
json_file_path = os.path.join(experiment_path, "params.json")

# Open the JSON file and load it into a dictionary
with open(json_file_path, 'r') as file:
    config = json.load(file)

# Instantiate the model
model = DynamicSemanticRelationModel(
    input_size=768*2,
    num_classes=4,
    hidden_sizes=config["hidden_sizes"],
    activation=config["activation"],
    dropout_rate=config["dropout_rate"]
).to(device)

# Load the model weights
pth_files = glob.glob(os.path.join(experiment_path, '**', '*.pth'), recursive=True)
model.load_state_dict(torch.load(pth_files[0], map_location=device))

df = pd.read_csv("Production/Generate candidates/candidate_files_biezi.csv")

df["sense1_gloss_embedding"] = df["sense1_gloss_embedding"].apply(deserialize_tensor)
df["sense2_gloss_embedding"] = df["sense2_gloss_embedding"].apply(deserialize_tensor)

print('Model weights loaded.')

concatenated_vectors = df.apply(
    lambda row: torch.cat((row['sense1_gloss_embedding'].flatten(), row['sense2_gloss_embedding'].flatten())),
    axis=1
)
# Stack all concatenated vectors into a single tensor
X = torch.stack(concatenated_vectors.tolist())

# Move input tensor to the same device as the model
X = X.to(device)

# Use the model to make predictions
model.eval()
with torch.no_grad():
    predictions = model(X)
    class_labels = ["hypernym", "hyponym", "none", "synonymy"]
    tensor_cpu = predictions.cpu()
    tensor_np = tensor_cpu.numpy()
    if len(df) != tensor_np.shape[0]:
        raise ValueError("Number of rows in DataFrame and tensor do not match.")
    # Add each column with None as the default value
    for i, col in enumerate(class_labels):
        df[col] = tensor_np[:, i]

df = df.drop(columns=["sense1_gloss_embedding","sense2_gloss_embedding"])
df.to_csv("Production/Generate candidates/results_example_biezi.csv",encoding='utf-8')


        

        # Example DataFrame with a column to store predictions
       

    


    
