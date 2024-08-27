import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
import pickle
from tqdm import tqdm 
from ray import tune, train
from ray.tune import grid_search
import ray
import os
import tempfile
from ray.train import Checkpoint
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(percentage_of_postive):
    df = pd.read_csv(r'C:\Users\davis\OneDrive\Desktop\Eliza\ai.lab_programming\Production\training_dataset_nouns.csv',usecols=['sense1_gloss_embedding', "sense2_gloss_embedding", "rel_type"])
    df = df[~df['rel_type'].isin(['holonym', 'meronym', 'antonym', 'also', 'similar'])]
    count_non_none = df[~df['rel_type'].str.contains("none")].shape[0]
    total_rows = round(count_non_none / percentage_of_postive)
    count_none_grandparents = df[df['rel_type'] == 'none_grandparents'].shape[0]
    number_of_random_rows = total_rows- count_none_grandparents
    none_random_df = df[df['rel_type'] == 'none_random']
    selected_random = none_random_df.head(round(number_of_random_rows*0.25))
    none_similarity_df = df[df['rel_type'] == 'none_similarity']
    selected_similarity = none_similarity_df.head(round(number_of_random_rows*0.75))
    df = pd.concat([selected_similarity,selected_random, df[~df['rel_type'].isin(['none_random', 'none_similarity'])]])
    df["sense1_gloss_embedding"] = df["sense1_gloss_embedding"].apply(deserialize_tensor)
    df["sense2_gloss_embedding"] = df["sense2_gloss_embedding"].apply(deserialize_tensor) 
    df['rel_type'] = df['rel_type'].apply(lambda x: 'none' if 'none' in x else x)
    return df

def train_model(config):

    df = get_data(percentage_of_postive=config["postive_percentage"])

    concatenated_vectors = df.apply(
        lambda row: torch.cat((row['sense1_gloss_embedding'].flatten(), row['sense2_gloss_embedding'].flatten())),
        axis=1
    )
    # Stack all concatenated vectors into a single tensor
    X = torch.stack(concatenated_vectors.tolist())
    print(X.shape)
    
    df['rel_type'] = pd.Categorical(df['rel_type'])
    y = torch.tensor(df['rel_type'].cat.codes.values, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    model = DynamicSemanticRelationModel(
        input_size=768*2,  # Assuming 768 is the size of each word embedding
        num_classes=len(df["rel_type"].unique()),    # Adjust to the actual number of classes
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        dropout_rate=config["dropout_rate"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])

    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]

    for epoch in (range(num_epochs)):
        print(epoch, "/", num_epochs)
        # Shuffle the training data
        permutation = torch.randperm(X_train.size()[0])
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        # Mini-batch training
        for i in (range(0, X_train.size(0), batch_size)):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]  # batch_y should be of shape [batch_size]

            optimizer.zero_grad()

            outputs = model(batch_X)  # Outputs logits

            # Compute the loss with CrossEntropyLoss
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()




    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted_classes = torch.max(test_outputs, 1)
        
        y_test_cpu = y_test.cpu()
        predicted_classes_cpu = predicted_classes.cpu()

        # Calculate metrics
        accuracy = accuracy_score(y_test_cpu, predicted_classes_cpu)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_cpu, predicted_classes_cpu, average='weighted')
        conf_matrix = confusion_matrix(y_test_cpu, predicted_classes_cpu)
        class_report = classification_report(y_test_cpu, predicted_classes_cpu, target_names=df['rel_type'].cat.categories)

        # Report metrics to Ray Tune
        report = {
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1_score":f1,
            "confusion_matrix":conf_matrix,
            "classification_report":class_report,
        }

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        torch.save(
            model.state_dict(),
            os.path.join(temp_checkpoint_dir, f"weights.pth"),
        )
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        train.report(report, checkpoint=checkpoint)
    



    # Save the model after training



# Define the search space
search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),  # Replace loguniform with grid search
    "num_epochs": tune.randint(40,70),
    "batch_size": tune.choice([32, 64]),
    "hidden_sizes": tune.choice([
        [1024, 512, 256, 128, 64]
    ]),
    "activation": tune.choice(['relu']),
    "optimizer": tune.choice(['Adam']),
    "dropout_rate": tune.choice([0.0]),
    "postive_percentage" : tune.choice([0.4,0.5,0.6,0.7])
}


# Run the grid search
def dynamic_trial_name_creator(trial):
    return f"run{trial.trial_id}"


ray.init(num_cpus=12, num_gpus=1)
analysis = tune.run(
    train_model, # Your training function
    config=search_space,
    trial_dirname_creator=dynamic_trial_name_creator,
    storage_path="C:/ray_results",
    resources_per_trial={"cpu": 12, "gpu": 1},
    num_samples=100
)

# Print the best result
print("Best config: ", analysis.get_best_config(metric="accuracy", mode="max"))

# Optionally print out the detailed report of the best trial
best_trial = analysis.get_best_trial(metric="accuracy", mode="max")
print("Best trial final accuracy: {}".format(best_trial.last_result["accuracy"]))
print("Best trial final precision: {}".format(best_trial.last_result["precision"]))
print("Best trial final recall: {}".format(best_trial.last_result["recall"]))
print("Best trial final F1 score: {}".format(best_trial.last_result["f1_score"]))
print("Best trial confusion matrix:\n", best_trial.last_result["confusion_matrix"])
print("Best trial classification report:\n", best_trial.last_result["classification_report"])

