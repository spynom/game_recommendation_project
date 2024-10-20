import torch
from torch import nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import json

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set randomness for reproducibility
np.random.seed(34)
torch.manual_seed(57)


# Define the NHybF architecture
class NHybF(nn.Module):
    def __init__(self, num_items, num_users, items_embedding_dim, users_embedding_dim, gmf_embedding_dim,
                 hidden_layers_size=(50, 50, 50, 50), dropout_prob=(0.25, 0.25, 0.25, 0.25), output_size=1):
        super(NHybF, self).__init__()

        # MLP architecture
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=items_embedding_dim)
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=users_embedding_dim)

        # Hidden layers with dropout
        self.hidden_layer1 = nn.Linear(in_features=items_embedding_dim + users_embedding_dim + 358,
                                       out_features=hidden_layers_size[0])
        self.dropout1 = nn.Dropout(dropout_prob[0])

        self.hidden_layer2 = nn.Linear(in_features=hidden_layers_size[0], out_features=hidden_layers_size[1])
        self.dropout2 = nn.Dropout(dropout_prob[1])

        self.hidden_layer3 = nn.Linear(in_features=hidden_layers_size[1], out_features=hidden_layers_size[2])
        self.dropout3 = nn.Dropout(dropout_prob[2])

        self.hidden_layer4 = nn.Linear(in_features=hidden_layers_size[2], out_features=hidden_layers_size[3])
        self.dropout4 = nn.Dropout(dropout_prob[3])

        # Output layer for MLP
        self.HybMLP_output_layer = nn.Linear(in_features=hidden_layers_size[3], out_features=output_size)

        # GMF architecture
        self.items_embedding1 = nn.Embedding(num_items, gmf_embedding_dim)
        self.users_embedding1 = nn.Embedding(num_users, gmf_embedding_dim)
        self.gmf_dropout = nn.Dropout(dropout_prob[4])

        # Final output layer combining MLP and GMF
        self.output_layer = nn.Linear(in_features=output_size + gmf_embedding_dim, out_features=1)

    def forward(self, x1, x2, x3):
        # MLP forward pass
        item_vector = self.item_embedding(x1.reshape(-1, 1)).squeeze(1)
        user_vector = self.user_embedding(x2.reshape(-1, 1)).squeeze(1)
        concatenation = torch.cat((item_vector, user_vector, x3), dim=1)

        relu1 = F.relu(self.dropout1(self.hidden_layer1(concatenation)))
        relu2 = F.relu(self.dropout2(self.hidden_layer2(relu1)))
        relu3 = F.relu(self.dropout3(self.hidden_layer3(relu2)))
        relu4 = F.relu(self.dropout4(self.hidden_layer4(relu3)))

        HybMLP_output = self.HybMLP_output_layer(relu4)

        # GMF forward pass
        GMF_item_vector = self.items_embedding1(x1)
        GMF_user_vector = self.users_embedding1(x2)
        GMF_mul_vector = GMF_item_vector * GMF_user_vector
        GMF_output = self.gmf_dropout(GMF_mul_vector)

        # Final concatenation and output
        concatenation = torch.cat((HybMLP_output, GMF_output), dim=1)
        return torch.sigmoid(self.output_layer(concatenation))


# Initialize model
NHybF_model = NHybF(num_items=227, num_users=666536, items_embedding_dim=20, users_embedding_dim=40,
                    gmf_embedding_dim=20, output_size=15, hidden_layers_size=(128, 16, 128, 32),
                    dropout_prob=(0.75, 0.5, 0.75, 0.5, 0.5)).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(NHybF_model.parameters(), lr=0.001)

# Load training data
class_0 = torch.from_numpy(np.load("../data/train_class_0.npy")).type(torch.float32)
class_1 = torch.from_numpy(np.load("../data/train_class_1.npy")).type(torch.float32)

# Load validation data
val_class_0 = torch.from_numpy(np.load("../data/val_class_0.npy")).type(torch.float32)
val_class_1 = torch.from_numpy(np.load("../data/val_class_1.npy")).type(torch.float32)
val = torch.from_numpy(np.concatenate([val_class_0, val_class_1], axis=0)).type(torch.float32).to(device)

# Training parameters
epoch_size = 8
batch_size = 10000
no_of_batches = class_1.shape[0] // batch_size
class_0_no_of_batches = class_0.shape[0] // batch_size

# Report dictionary to store losses
report = {"train_loss": [], "val_loss": []}

# Training loop
for epoch in range(epoch_size):
    NHybF_model.to(device)
    j = 0

    # Shuffle indices for class 0 and 1
    class_0_index = np.arange(class_0.shape[0])
    class_1_index = np.arange(class_1.shape[0])

    for i in range(no_of_batches):
        NHybF_model.train()

        # Prepare batch data
        class_0_batch_index = class_0_index[j * batch_size:(j + 1) * batch_size]
        class_1_batch_index = class_1_index[i * batch_size:(i + 1) * batch_size]

        x1 = torch.concatenate((class_0[class_0_batch_index, 0], class_1[class_1_batch_index, 0]), 0).to(device).to(
            torch.int32)
        x2 = torch.concatenate((class_0[class_0_batch_index, 1], class_1[class_1_batch_index, 1]), 0).to(device).to(
            torch.int32)
        x3 = torch.concatenate((class_0[class_0_batch_index, 2:-1], class_1[class_1_batch_index, 2:-1]), 0).to(device)

        labels = torch.concatenate((class_0[class_0_batch_index, -1], class_1[class_1_batch_index, -1]), 0).reshape(-1,
                                                                                                                    1).to(
            device)

        # Zero gradients, forward pass, loss calculation, backward pass, and optimizer step
        optimizer.zero_grad()
        output = NHybF_model(x1, x2, x3)
        train_loss = criterion(output, labels)
        train_loss.backward()
        optimizer.step()

        # Validation loss calculation
        if not i % 10:
            NHybF_model.eval()
            with torch.no_grad():
                x1 = val[:, 0].to(torch.int32)
                x2 = val[:, 1].to(torch.int32)
                x3 = val[:, 2:-1]
                val_labels = val[:, -1].reshape(-1, 1)

                val_output = NHybF_model(x1, x2, x3)
                val_loss = criterion(val_output, val_labels)

            print(
                f'step: {i}/{no_of_batches}, epoch: {epoch + 1}, Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        # Update index for class 0
        j += 1 if j % class_0_no_of_batches else 0
        np.random.shuffle(class_0_index)

        # Early stopping criteria
        if val_loss < 0.525 and train_loss < 0.525 and epoch > 6:
            break

    np.random.shuffle(class_1_index)

    # Final evaluation on the entire training set
    x1 = torch.concatenate((class_0[:, 0], class_1[:, 0]), 0).to(torch.int32)
    x2 = torch.concatenate((class_0[:, 1], class_1[:, 1]), 0).to(torch.int32)
    x3 = torch.concatenate((class_0[:, 2:-1], class_1[:, 2:-1]), 0)

    labels = torch.concatenate((class_0[:, -1], class_1[:, -1]), 0).reshape(-1, 1)

    NHybF_model.to("cpu").eval()
    with torch.no_grad():
        output = NHybF_model(x1, x2, x3)
        train_loss = criterion(output, labels)
        report["train_loss"].append(train_loss.item())
        report["val_loss"].append(val_loss.item())

# Save the model state
torch.save(NHybF_model.state_dict(), 'models/NHybF_model.pth')

# Save training report
with open('reports/NHybF_train_report.json', 'w') as f:
    json.dump(report, f)
