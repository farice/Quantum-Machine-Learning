import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

from scipy.stats import unitary_group
import matplotlib.pyplot as plt

import itertools
from functools import reduce

N = 2

dev = qml.device('default.qubit', wires=N)

num_qubits = N
num_layers = 4
var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)

num_iterations = 100

def U_phi(x):
    # x_2 = (pi - x_0)(pi - x_1)
    for i in range(N):
        qml.RZ( x[i], wires=0)


    for (j, pair) in enumerate(itertools.combinations(range(N), r=2)):
        qml.CNOT(wires=[pair[0], pair[1]])
        qml.RZ( x[N + j], pair[1])
        qml.CNOT(wires=[pair[0], pair[1]])

def featuremap(x):
    for i in range(layers):
        for j in range(N):
            qml.Hadamard(wires=j)
        U_phi(x)

def layer(W): # 6 weights are specified at each layer

    for i in range(N):
        if i == (N-1):
            qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
            qml.Rot(W[N-1, 0], W[N-1, 1], W[N-1, 2], wires=N-1)

            qml.CNOT(wires=[0, N-1])
        else:
            # euler angles
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
            qml.Rot(W[i+1, 0], W[i+1, 1], W[i+1, 2], wires=i + 1)

            qml.CNOT(wires=[i, i+1])


@qml.qnode(dev)
def circuit(weights, x, n=0):

    featuremap(x)

    for W in weights:
        layer(W)

    return qml.expval.PauliZ(wires=n)

def variational_classifier(var, x): # x is a keyword argument -> fixed (not trained)
    weights = var[0]
    bias = var[1]

    exp_Z = circuit(weights, x, n=0)
    for i in range(1, N):
        e = circuit(weights,x,n=i)
        exp_Z *= e

    return exp_Z + bias


def square_loss(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)

    return loss


def accuracy(labels, predictions):
    #print(labels, predictions)
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


def cost(var, X, Y):

    predictions = [variational_classifier(var, x) for x in X]
    #if (len(Y) == num_data):
    #    print("[(pred, label), ...]: ", list(zip(predictions, Y)))
    return square_loss(Y, predictions)

def gen_random_U():
  random_U = unitary_group.rvs(2 ** N)
  random_U = random_U / (np.linalg.det(random_U) ** (1/(2**N))) # so that det = 1


  return random_U

@qml.qnode(dev)
def data_label(x, i=0):
    #print(u)
    #print("label the following:", x)
    featuremap(x)
    qml.QubitUnitary(random_U, wires=list(range(N)))

    return qml.expval.PauliZ(wires=i)

def gen_data(thresh):
    #thresh = 0.3

    X = np.array([])
    Y = np.array([])
    ctr = 0 # num valid data pts
    maxval = 0.0
    minval = 0.0

    np.random.seed(0)

    while ctr < 40:
        x = np.random.rand(N) * 2 * np.pi
        for pair in itertools.combinations(range(N), r=2):
            x = np.append(x, (np.pi - x[pair[0]]) * (np.pi - x[pair[1]]))

        y = []
        for i in range(N):
            y.append(data_label(x, i=i))

        y_prod = reduce((lambda x, y: x * y), y)

        #print(y, y_prod)
        if (y_prod > maxval):
            maxval = y_prod
            print("new max separation: ", maxval)
        elif (y_prod < minval):
            minval = y_prod
            print("new min separation: ", minval)

        if y_prod > thresh:
            Y = np.append(Y, +1)
            X = np.append(X, x)
            ctr += 1
            #print("+1")
        elif y_prod < -1 * thresh:
            Y = np.append(Y, -1)
            X = np.append(X, x)
            ctr += 1
            #print("-1")

    X = X.reshape(-1, 3)
    print("Data: ", list(zip(X, Y)))
    return X, Y

def divide_train_test(X, Y):
    global num_data
    num_data = len(Y)
    global num_train
    num_train = int(0.5 * num_data)

    print("size data, size train: ", num_data, num_train)

    index = np.random.permutation(range(num_data))
    X_train = X[index[:num_train]]
    Y_train = Y[index[:num_train]]

    X_test = X[index[num_train:]]
    Y_test = Y[index[num_train:]]

    return X_train, Y_train, X_test, Y_test

def train_and_test(X_train, Y_train, X_test, Y_test):
    opt = NesterovMomentumOptimizer(0.01)
    batch_size = 5

    # train the variational classifier
    var = var_init

    test_accuracies = []
    train_accuracies = []
    costs = []
    for it in range(num_iterations):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, num_train, (batch_size, ))
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        var = opt.step(lambda v: cost(v, X_train_batch, Y_train_batch), var)

        # Compute predictions on train and validation set
        predictions_train = [np.sign(variational_classifier(var, f)) for f in X_train]
        predictions_test = [np.sign(variational_classifier(var, f)) for f in X_test]

        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_train, predictions_train)
        acc_test = accuracy(Y_test, predictions_test)

        # Compute cost on all samples
        c = cost(var, X, Y)

        costs.append(c)
        test_accuracies.append(acc_test)
        train_accuracies.append(acc_train)

        print("Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
              "".format(it+1, c, acc_train, acc_test))

    return train_accuracies, test_accuracies, costs, var

def main():
    
