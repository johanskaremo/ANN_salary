import pandas as pd
from dataCleaning import cleanData
import numpy as np
from model.engine import Value
from model.ann import Neuron, Layer, MLP
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#retrieve current batch
def get_batch(X, y, batch_index, batch_size):
    start = batch_index * batch_size
    end = start + batch_size
    return X[start:end], y[start:end]

#Needed for adressing gradient explosion
def clip_gradients(model, max_value):
    for param in model.parameters():
        param.grad = np.clip(param.grad, -max_value, max_value)

def crossEntropyLoss(y_true, y_pred):
    # y_true is a list of integers representing the correct classes
    # and y_pred is a list of lists of Value objects (softmax probabilities).
    loss = 0
    for yi, pi in zip(y_true, y_pred):
        # Get the probability corresponding to the true class
        p = pi[yi]
        # Cross-entropy loss for this instance
        loss_instance = -p.log()
        loss += loss_instance
    return loss / len(y_true)

def run_epoch(X_data, y_data, batch_size, model, learning_rate):    #newly added
    tot_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = len(X_data) // batch_size

    for i in range(num_batches):
        Xb, yb = get_batch(X_data, y_data, i, batch_size)   #retrieve current batch
        inputs = [list(map(Value, xrow)) for xrow in Xb]    #convert all inputs to Value objects

        #Forward pass
        scores = list(map(model, inputs))
        data_loss = crossEntropyLoss(yb, scores)    #calculate loss for batch

        if model.training:  # Check if model is in training mode
            model.zero_grad()
            #backward pass
            data_loss.backward()
            clip_gradients(model, max_value=1.5)
            for p in model.parameters():
                p.data -= learning_rate * p.grad    #tweak weighs & biases, gradient descent

        tot_loss += data_loss.data  #add current batch loss to total loss of epoch
        batch_accuracy = sum(yi == np.argmax([p.data for p in scorei]) for yi, scorei in zip(yb, scores))
        total_correct += batch_accuracy
        total_samples += len(yb)

    epoch_loss = tot_loss / num_batches
    epoch_accuracy = total_correct / total_samples
    return epoch_loss, epoch_accuracy * 100


#Start of program, read dataset and split in train and validation set
data = pd.read_csv("salaries.csv")
df = cleanData(data)
# #Convert dataframe to numpy arrays
y = df["salary_in_usd"]
y = y.to_numpy()
df = df.drop(columns=["salary_in_usd"])
X = df.to_numpy()
X = X.astype(np.float16)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

#Initialization of Neural net
model = MLP(5,[8,8,5]) # 2-layer neural network

#Hyperparameters
batch_size = 12
num_epochs = 10
learning_rate = 0.0005

#for graphing the losses
traini_loss = []

vali_loss = []

for epoch in range(num_epochs):
    # Training Phase
    model.train()  # Set the model to training mode if your model differentiates
    train_loss, train_accuracy = run_epoch(X_train, y_train, batch_size, model, learning_rate)
    traini_loss.append(train_loss)

    # Validation Phase
    model.eval()  # Set the model to evaluation mode
    val_loss, val_accuracy = run_epoch(X_val, y_val, batch_size, model, learning_rate)
    vali_loss.append(val_loss)
    print(f"Epoch {epoch}: Train Loss {train_loss}, Train Accuracy {train_accuracy}%, "
          f"Val Loss {val_loss}, Val Accuracy {val_accuracy}%")


#For visualization
#epochs = range(1, num_epochs + 1)

# # Plotting the train and validation loss
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, traini_loss, color='blue', label='Training Loss')
# plt.plot(epochs, vali_loss, color='red', label='Validation Loss')
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()