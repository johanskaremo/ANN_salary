import pandas as pd
from dataCleaning import cleanData
import numpy as np
from model.engine import Value
from model.ann import Neuron, Layer, MLP
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_moons, make_blobs

def clip_gradients(model, max_value):
    for param in model.parameters():
        param.grad = np.clip(param.grad, -max_value, max_value)

def crossEntropyLoss(y_true, y_pred):
    # Assuming y_true is a list of integers representing the correct classes
    # and y_pred is a list of lists of Value objects (softmax probabilities).
    loss = 0
    for yi, pi in zip(y_true, y_pred):
        # Get the probability corresponding to the true class
        p = pi[yi]
        # Cross-entropy loss for this instance
        loss_instance = -p.log()
        loss += loss_instance
    #return loss / Value(len(y_true))
    return loss / len(y_true)

def loss(batch_size):
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]

    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model to get scores
    scores = list(map(model, inputs))

    # # svm "max-margin" loss
    # losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    # data_loss = sum(losses) * (1.0 / len(losses))
    # # L2 regularization
    # alpha = 1e-4
    # reg_loss = alpha * sum((p * p for p in model.parameters()))
    # total_loss = data_loss + reg_loss

    tot_loss = crossEntropyLoss(yb, scores)

    # also get accuracy
    accuracy = [yi == np.argmax([p.data for p in scorei]) for yi, scorei in zip(yb, scores)]
    return tot_loss, sum(accuracy) / len(accuracy)

data = pd.read_csv("salaries.csv")
df = cleanData(data)

# #Convert dataframe to numpy arrays
y = df["salary_in_usd"]
y = y.to_numpy()
#y = np.expand_dims(y, axis=1)
df = df.drop(columns=["salary_in_usd"])
X = df.to_numpy()
X = X.astype(np.float16)
X = X[-10:]    #ta bort sen
y = y[-10:]    #ta bort sen


#y = np.argmax(y, axis=-1)
# X = X.tolist()
# y = y_new.tolist()
#Initialize neural net


# np.random.seed(1337)
# random.seed(1337)
# X, y = make_moons(n_samples=100, noise=0.1)
#
# y = y*2 - 1 # make y be -1 or 1

#model = MLP(5, [16, 16, 5]) # 2-layer neural network orginal
model = MLP(5, [16, 16, 5]) # 2-layer neural network
num_epochs = 100
num_samples = X.shape[0]
batch_size = 4
num_batches = num_samples // batch_size

for epoch in range(num_epochs):  #100 från början, detta är den ursprungliga trainingloop
    tot_loss = 0
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        # forward
        batch_loss, batch_acc = loss(batch_size = batch_size)

        # backward
        model.zero_grad()
        batch_loss.backward()
        clip_gradients(model, max_value=1.0)
        # update (sgd)
        learning_rate = 1.0 - 0.9 * epoch / num_epochs
        for p in model.parameters():
            p.data -= learning_rate * p.grad
            #print(p.grad)
        tot_loss += batch_loss.data
        total_correct += batch_acc * batch_size
        total_samples += batch_size

    epoch_loss = tot_loss / num_batches
    epoch_accuracy = total_correct / total_samples
    print(f"Epoch {epoch}: Loss {epoch_loss}, Accuracy {epoch_accuracy * 100}%")
# for k in range(100):  #100 från början, detta är den ursprungliga trainingloop
#     # forward
#     total_loss, acc = loss()
#
#     # backward
#     model.zero_grad()
#     total_loss.backward()
#     clip_gradients(model, max_value=1.0)
#     # update (sgd)
#     learning_rate = 1.0 - 0.9 * k / 100
#     for p in model.parameters():
#         p.data -= learning_rate * p.grad
#         #print(p.grad)
#
#     if k % 1 == 0:
#         print(f"epoch {k} loss {total_loss.data}, accuracy {acc * 100}%")
#

