# import libraries
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
# check cuda availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set randomness for reproducibility
np.random.seed(34)
torch.manual_seed(57)

# architecture of HybMLP
class HybMLP(nn.Module):
    def __init__(self, num_items,num_users, items_embedding_dim,users_embedding_dim,hidden_layers_size=(50,50,50,50),dropout_prob=(0.25,0.25,0.25),output_size=1):
        super(HybMLP,self).__init__()

        self.item_embedding = nn.Embedding(num_embeddings=num_items,embedding_dim=items_embedding_dim)# defining items embedding layer

        self.user_embedding = nn.Embedding(num_embeddings=num_users,embedding_dim=users_embedding_dim) # defining users embedding layer

        self.hidden_layer1 = nn.Linear(in_features=items_embedding_dim+users_embedding_dim+358, out_features=hidden_layers_size[0]) # hidden layer 1
        self.dropout1 = nn.Dropout(dropout_prob[0]) # dropout 1

        self.hidden_layer2=nn.Linear(in_features=hidden_layers_size[0],out_features=hidden_layers_size[1]) # hidden layer 2
        self.dropout2 = nn.Dropout(dropout_prob[1])  # dropout 2

        self.hidden_layer3=nn.Linear(in_features=hidden_layers_size[1],out_features=hidden_layers_size[2]) # hidden layer 3
        self.dropout3 = nn.Dropout(dropout_prob[2]) # dropout 3

        self.hidden_layer4=nn.Linear(in_features=hidden_layers_size[2],out_features=hidden_layers_size[3]) # hidden layer 4
        self.dropout4 = nn.Dropout(dropout_prob[3]) # dropout 4

        self.output_layer = nn.Linear(in_features=hidden_layers_size[3],out_features=output_size) # output layer

    def forward(self,x1,x2,x3):

        item_vector=self.item_embedding(x1) # applying embedding on x1 (item)

        user_vector=self.user_embedding(x2) # applying embedding on x2 (user)

        concatenation=torch.cat((item_vector, user_vector, x3), dim=1) # concatenating item vector, user vector and x3 (included item information and user information)

        # hidden layer 1
        first_hidden=self.hidden_layer1(concatenation) # passing concatenating matrix to first hidden layer
        dropout_output1=self.dropout1(first_hidden) # applying dropout on first hidden layer
        relu1=F.relu(dropout_output1) # applying relu activation function

        # hidden layer 2
        second_hidden=self.hidden_layer2(relu1) # passing hidden layer1 output to second hidden layer
        dropout_output2=self.dropout2(second_hidden) # applying dropout on second hidden layer
        relu2=F.relu(dropout_output2) # applying relu activation function

        # hidden layer 3
        third_hidden=self.hidden_layer3(relu2) # passing hidden layer2 output to third hidden layer
        dropout_output3=self.dropout3(third_hidden) # applying dropout on third hidden layer
        relu3=F.relu(dropout_output3) # applying relu activation function

        # hidden layer 4
        forth_hidden=self.hidden_layer4(relu3) # passing hidden layer3 output to forth hidden layer
        dropout_output4=self.dropout4(forth_hidden) # applying dropout on forth hidden layer
        relu4=F.relu(dropout_output4) # applying relu activation function

        # output layer
        output=self.output_layer(relu4)  # passing hidden layer4 output to output layer
        return torch.sigmoid(output) # applying sigmoid activation function

# initializing model, loss function and optimizer
HybMLP_model=HybMLP(num_items=227,num_users=666536,items_embedding_dim= 20,users_embedding_dim=100,hidden_layers_size=(64,132,16,6),dropout_prob=(0.75,0.75,0.5,0.75)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(HybMLP_model.parameters(), lr=0.001)


class_0=np.load("data/train_class_0.npy") # loading class 0 contain train file
class_0=torch.from_numpy(class_0).type(torch.float32) # converting into torch tensor
class_1=np.load("data/train_class_1.npy") # loading class 1 contain train file
class_1=torch.from_numpy(class_1).type(torch.float32) # converting into torch tensor


val_class_0=np.load("data/val_class_0.npy") # loading class 0 contain validation file
val_class_1=np.load("data/val_class_1.npy")  # loading class 1 contain validation file
val=torch.from_numpy(np.concatenate([val_class_0,val_class_1],axis=0)).type(torch.float32).to(device) # merging and converting into tensor


epoch_size=11 # no of epoch
batch_size=10000 # size of batch
no_of_batches=class_1.shape[0]//batch_size # no of batches
class_0_no_of_batches=class_0.shape[0]//batch_size # separate no of batches for class 0 to handel imbalance

report={"train_loss":[],"val_loss":[]} # dic to save report of training
for epoch in range(epoch_size):
    HybMLP_model.to(device)
    j=0 # current the batch no. for class 0
    class_0_index=np.arange(class_0.shape[0]) # indexes of class 0
    class_1_index=np.arange(class_1.shape[0]) # indexes of class 1
    for i in range(no_of_batches):
        HybMLP_model.train() # change mode of model to train
        class_0_batch_index=class_0_index[j*batch_size:(j+1)*batch_size]
        class_1_batch_index=class_1_index[i*batch_size:(i+1)*batch_size]
        x1=torch.concatenate((
            class_0[class_0_batch_index,0],
            class_1[class_1_batch_index,0],
        ),0).to(device).to(torch.int32) # game id input
        x2=torch.concatenate((
            class_0[class_0_batch_index,1],
            class_1[class_1_batch_index,1],
        ),0).to(device).to(torch.int32) # user id input
        x3=torch.concatenate((
            class_0[class_0_batch_index,2:-1],
            class_1[class_1_batch_index,2:-1],
        ),0).to(device) # games and users additional information

        label1=class_0[class_0_batch_index,-1]
        label2=class_1[class_1_batch_index,-1]
        labels=torch.concatenate((label1,label2),0).to(device)
        labels=labels.reshape(-1,1) # target attribute

        optimizer.zero_grad() # reset gradient of optimizer
        output=HybMLP_model(x1,x2,x3) # y_hat
        train_loss=criterion(output,labels) # calculate loss
        train_loss.backward() # applying backward propagation
        optimizer.step() # update parameters

        if not i%10:
            HybMLP_model.eval() # change model mode evaluation
            x1=val[:,0].to(torch.int32)
            x2=val[:,1].to(torch.int32)
            x3=val[:,2:-1]
            labels=val[:,-1]
            labels=labels.reshape(-1,1)
            with torch.no_grad():
                output=HybMLP_model(x1,x2,x3) # val y_hat
                val_loss=criterion(output,labels) # val loss
            print(f'step:{i}/{no_of_batches},epoch{epoch+1}, Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        if j%class_0_no_of_batches:
            j+=1 # updating j batch no of class 0
        else:
            np.random.shuffle(class_0_index) # add randomness in index of class 0 instances
            j=0 # reset batch no of class 0


    np.random.shuffle(class_1_index) # add randomness in index of class 1 instances

    # evaluating model performance on overall train dataset
    x1=torch.concatenate((
        class_0[:,0],
        class_1[:,0],
    ),0).to(torch.int32)
    x2=torch.concatenate((
        class_0[:,1],
        class_1[:,1],
    ),0).to(torch.int32)
    x3=torch.concatenate((
        class_0[:,2:-1],
        class_1[:,2:-1],
    ),0)

    label1=class_0[:,-1]
    label2=class_1[:,-1]
    labels=torch.concatenate((label1,label2),0).reshape(-1,1)

    HybMLP_model.to("cpu").eval()
    with torch.no_grad():
        output=HybMLP_model(x1,x2,x3)
        train_loss=criterion(output,labels)
        report["train_loss"].append(train_loss.item())
        report["val_loss"].append(val_loss.item())


torch.save(HybMLP_model.state_dict(), 'models/HybMLP_model.pth') # saving state of model

with open('reports/HyMLP_train_report.json', 'w') as f:
    json.dump(report,f) # saving training report as json file
