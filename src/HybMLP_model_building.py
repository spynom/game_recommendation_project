# Import libraries
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import json
import torch.nn.functional as F

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set randomness for reproducibility
np.random.seed(34)
torch.manual_seed(57)


# Architecture of HybMLP
class HybMLP(nn.Module):
    def __init__(self, num_items, num_users, items_embedding_dim, users_embedding_dim,
                 hidden_layers_size=(50, 50, 50, 50), dropout_prob=(0.25, 0.25, 0.25), output_size=1):
        super(HybMLP, self).__init__()

        # Define item and user embedding layers
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=items_embedding_dim)
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=users_embedding_dim)

        # Define hidden layers and corresponding dropout
        self.hidden_layer1 = nn.Linear(in_features=items_embedding_dim + users_embedding_dim + 358,
                                       out_features=hidden_layers_size[0])
        self.dropout1 = nn.Dropout(dropout_prob[0])

        self.hidden_layer2 = nn.Linear(in_features=hidden_layers_size[0], out_features=hidden_layers_size[1])
        self.dropout2 = nn.Dropout(dropout_prob[1])

        self.hidden_layer3 = nn.Linear(in_features=hidden_layers_size[1], out_features=hidden_layers_size[2])
        self.dropout3 = nn.Dropout(dropout_prob[2])

        self.hidden_layer4 = nn.Linear(in_features=hidden_layers_size[2], out_features=hidden_layers_size[3])
        self.dropout4 = nn.Dropout(dropout_prob[3])

        # Output layer
        self.output_layer = nn.Linear(in_features=hidden_layers_size[3], out_features=output_size)

    def forward(self, x1, x2, x3):
        # Apply embeddings
        item_vector = self.item_embedding(x1.reshape(-1, 1)).squeeze(1)
        user_vector = self.user_embedding(x2.reshape(-1, 1)).squeeze(1)

        # Concatenate item, user, and additional features
        concatenation = torch.cat((item_vector, user_vector, x3), dim=1)

        # Pass through hidden layers with dropout and ReLU activation
        first_hidden = self.hidden_layer1(concatenation)
        dropout_output1 = self.dropout1(first_hidden)
        relu1 = F.relu(dropout_output1)

        second_hidden = self.hidden_layer2(relu1)
        dropout_output2 = self.dropout2(second_hidden)
        relu2 = F.relu(dropout_output2)

        third_hidden = self.hidden_layer3(relu2)
        dropout_output3 = self.dropout3(third_hidden)
        relu3 = F.relu(dropout_output3)

        forth_hidden = self.hidden_layer4(relu3)
        dropout_output4 = self.dropout4(forth_hidden)
        relu4 = F.relu(dropout_output4)

        # Output layer
        output = self.output_layer(relu4)
        return torch.sigmoid(output)  # Sigmoid activation for binary classification


# Initialize model, loss function, and optimizer
HybMLP_model = HybMLP(num_items=227, num_users=666536, items_embedding_dim=20,
                      users_embedding_dim=100, hidden_layers_size=(64, 132, 16, 6),
                      dropout_prob=(0.75, 0.75, 0.5, 0.75)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(HybMLP_model.parameters(), lr=0.001)

# Load training data
class_0 = np.load("data/train_class_0.npy")
class_0 = torch.from_numpy(class_0).type(torch.float32)
class_1 = np.load("data/train_class_1.npy")
class_1 = torch.from_numpy(class_1).type(torch.float32)

# Load validation data
val_class_0 = np.load("data/val_class_0.npy")
val_class_1 = np.load("data/val_class_1.npy")
val = torch.from_numpy(np.concatenate([val_class_0, val_class_1], axis=0)).type(torch.float32).to(device)

# Training configuration
epoch_size = 11
batch_size = 10000
no_of_batches = class_1.shape[0] // batch_size
class_0_no_of_batches = class_0.shape[0] // batch_size

# Initialize report for losses
report = {"train_loss": [], "val_loss": []}

# Training loop
for epoch in range(epoch_size):
    HybMLP_model.to(device)
    j = 0  # Current batch number for class 0
    class_0_index = np.arange(class_0.shape[0])  # Indices for class 0
    class_1_index = np.arange(class_1.shape[0])  # Indices for class 1

    for i in range(no_of_batches):
        HybMLP_model.train()  # Set model to training mode

        # Prepare batch indices
        class_0_batch_index = class_0_index[j * batch_size:(j + 1) * batch_size]
        class_1_batch_index = class_1_index[i * batch_size:(i + 1) * batch_size]

        # Prepare inputs
        x1 = torch.cat((class_0[class_0_batch_index, 0], class_1[class_1_batch_index, 0]), 0).to(device).to(
            torch.int32)  # Game ID
        x2 = torch.cat((class_0[class_0_batch_index, 1], class_1[class_1_batch_index, 1]), 0).to(device).to(
            torch.int32)  # User ID
        x3 = torch.cat((class_0[class_0_batch_index, 2:-1], class_1[class_1_batch_index, 2:-1]), 0).to(
            device)  # Additional information

        # Prepare labels
        label1 = class_0[class_0_batch_index, -1]
        label2 = class_1[class_1_batch_index, -1]
        labels = torch.cat((label1, label2), 0).to(device).view(-1, 1)  # Target attribute

        optimizer.zero_grad()  # Reset gradients
        output = HybMLP_model(x1, x2, x3)  # Forward pass
        train_loss = criterion(output, labels)  # Calculate loss
        train_loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters

        # Evaluate and print validation loss every 10 steps
        if not i % 10:
            HybMLP_model.eval()  # Set model to evaluation mode
            x1 = val[:, 0].to(torch.int32)
            x2 = val[:, 1].to(torch.int32)
            x3 = val[:, 2:-1]
            labels = val[:, -1].view(-1, 1)
            with torch.no_grad():
                output = HybMLP_model(x1, x2, x3)  # Validation output
                val_loss = criterion(output, labels)  # Validation loss
            print(
                f'step:{i}/{no_of_batches}, epoch {epoch + 1}, Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        # Update class 0 batch index
        if j % class_0_no_of_batches:
            j += 1  # Increment class 0 batch index
        else:
            np.random.shuffle(class_0_index)  # Shuffle class 0 indices
            j = 0  # Reset batch number for class 0

    np.random.shuffle(class_1_index)  # Shuffle class 1 indices for next epoch

    # Evaluate model performance on the entire training dataset
    x1 = torch.cat((class_0[:, 0], class_1[:, 0]), 0).to(torch.int32)
    x2 = torch.cat((class_0[:, 1], class_1[:, 1]), 0).to(torch.int32)
    x3 = torch.cat((class_0[:, 2:-1], class_1[:, 2:-1]), 0)

    label1 = class_0[:, -1]
    label2 = class_1[:, -1]
    labels = torch.cat((label1, label2), 0).view(-1, 1)

    HybMLP_model.to("cpu").eval()
    with torch.no_grad():
        output = HybMLP_model(x1, x2, x3)  # Training output
        train_loss = criterion(output, labels)  # Training loss
        report["train_loss"].append(train_loss.item())  # Record training loss
        report["val_loss"].append(val_loss.item())  # Record validation loss

# Save model state
torch.save(HybMLP_model.state_dict(), 'models/HybMLP_model.pth')

# Save training report as JSON
with open('reports/HybMLP_train_report.json', 'w') as f:
    json.dump(report, f)
