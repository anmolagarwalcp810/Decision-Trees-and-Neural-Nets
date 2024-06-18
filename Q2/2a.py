import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--train',dest='train',required=False,default='poker/poker-hand-training-true.data',
                    help="Path to train file")
parser.add_argument('--test',dest='test',required=False,default='poker/poker-hand-testing.data',
                    help="Path to test file")
parser.add_argument('--adaptive',dest='adaptive',required=False,default=0,type=int,help='Adaptive learning rate, 0 for not adaptive, 1 for adaptive')
parser.add_argument('--activation',dest='activation',required=False,default='sigmoid',help='Activation function for hidden layer')
parser.add_argument('-q',dest='q',required=True,default='c',help='Question number')
parser.add_argument('-l1',dest='l1',required=False,default=10,type=int,help='Hidden Layer 1 units')
parser.add_argument('-l2',dest='l2',required=False,default=0,type=int,help='Hidden Layer 2 units')
parser.add_argument('-k',dest='k',required=False,default=30,type=int,help='Iterations in stopping criteria')
parser.add_argument('--delta',dest='delta',required=False,default=1e-7,type=float,help='Delta in stopping criteria')
args = parser.parse_args()

'arguments'
train_csv = args.train
test_csv = args.test
adaptive = bool(args.adaptive)
activation = args.activation
q = args.q
hidden_layers = [args.l1]
if args.l2>0:
    hidden_layers.append(args.l2)
print(hidden_layers)
k = args.k
delta = args.delta
'end'

'check'
print("Train_csv:",train_csv)
print("Test_csv:",test_csv)
print("Adaptive:",adaptive)
print("Activation:",activation)
print("Question part:",q)
print("Hidden Layers:",hidden_layers)
print("Iterations in Stopping Criteria:",k)
print("Delta in Stopping Criteria:",delta)
'end'

col_names = [str(i) for i in range(11)]
train_data = pd.read_csv(train_csv,delimiter=',',header=None,names=col_names)
test_data = pd.read_csv(test_csv,delimiter=',',header=None,names=col_names)

x_data_train = train_data.loc[:,train_data.columns!="10"]
x_data_test = test_data.loc[:,test_data.columns!="10"]

x_data_train = x_data_train.astype(str)
x_data_test = x_data_test.astype(str)

x_data_train = pd.get_dummies(x_data_train)
x_data_test = pd.get_dummies(x_data_test)

y_data_train = train_data.loc[:,train_data.columns=="10"]
y_data_test = test_data.loc[:,test_data.columns=="10"]

y_data_train = y_data_train.astype(str)
y_data_test = y_data_test.astype(str)

y_data_train = pd.get_dummies(y_data_train)
y_data_test = pd.get_dummies(y_data_test)

x_train = x_data_train.to_numpy()
x_test = x_data_test.to_numpy()

y_train = y_data_train.to_numpy()
y_test = y_data_test.to_numpy()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def d_relu(z):
    return (z>0).astype(float)

def relu(z):
    return z*d_relu(z)

def train_batch(M,layers,g,d_g,X,y,theta,lr):
    '''
    :param M: batch size
    :param layers: hidden layers
    :param g: activation
    :param d_g: derivation of activation
    :param X: training input : M x m
    :param y: training label : M x r
    :param theta: weights of the network
    :return: loss, and only updates theta, should get updated as passed by reference in list of theta's
    '''
    X = X.T         # m x M
    y = y.T         # r x M
    num_layers = len(layers)
    net = [None for _ in range(num_layers)]
    out = [None for _ in range(num_layers)]
    loss = 0
    out[0] = X
    # forward
    # across hidden layers
    for i in range(1,num_layers-1):
        net[i] = np.matmul(theta[i],out[i-1])
        out[i] = g(np.copy(net[i]))
    # num_layers-1 is output layer
    net[num_layers-1] = np.matmul(theta[num_layers-1],out[num_layers-2])
    out[num_layers-1] = sigmoid(net[num_layers-1])
    delta = [None for _ in range(num_layers)]
    # backward propagation
    # output
    delta[num_layers-1] = (y-out[num_layers-1])*d_sigmoid(net[num_layers-1])
    # hidden
    for i in range(num_layers-2,0,-1):
        delta[i] = np.matmul(theta[i+1].T,delta[i+1])*d_g(net[i])
    # update parameters
    for i in range(1,num_layers):
        theta[i] = theta[i]+lr*np.matmul(delta[i],out[i-1].T)
    loss = np.sum((y-out[num_layers-1])**2)/(2*M)
    return loss

def train(M,m,hidden_layers,r,g,d_g,X,y,orignial_lr,batch_size,k,delta,adaptive):
    layers = [m]
    layers.extend(hidden_layers)
    layers.append(r)
    num_layers = len(layers)
    theta = [None for _ in range(num_layers)]
    for i in range(1,num_layers):
        theta[i] = np.random.randn(layers[i],layers[i-1])*factor
    # print(theta)
    num_batchs = M//batch_size
    last_batch = M - num_batchs*batch_size
    e = 1
    iterations = 0
    previous_loss = -1000000
    current_loss = 0
    flag = True
    while(flag):
        # update learning rate if required
        if adaptive:
            lr = orignial_lr/np.sqrt(e)
            if lr<=min_lr:
                print("Minimum Learning Rate reached, stopping training")
                break
        else:
            lr = orignial_lr

        for b in range(num_batchs):
            # stopping criteria
            if iterations==k:
                if((abs(current_loss-previous_loss)/k)<=delta):
                    flag=True
                    break
                iterations = 0
                previous_loss = current_loss
                current_loss=0
            start = b*batch_size
            end = start+batch_size
            loss = train_batch(batch_size,layers,g,d_g,X[start:end],y[start:end],theta,lr)

            # if e%100==0: print("Loss: ",loss)

            iterations += 1
            current_loss+=loss

        # lass batch
        if iterations == k:
            if ((abs(current_loss - previous_loss) / k) <= delta):
                flag = True
                break
            iterations = 0
            previous_loss = current_loss
            current_loss=0
        loss = train_batch(last_batch,layers,g,d_g,X[(M-last_batch):M],y[(M-last_batch):M],theta,lr)
        iterations+=1
        current_loss += loss*last_batch/batch_size

        e+=1

    return theta, layers, e

def forward(M,layers,g,X,y,theta):
    X = X.T
    y = y.T
    num_layers = len(layers)
    net = [None for _ in range(num_layers)]
    out = [None for _ in range(num_layers)]
    loss = 0
    out[0] = X
    # forward
    # across hidden layers
    for i in range(1, num_layers - 1):
        net[i] = np.matmul(theta[i], out[i - 1])
        out[i] = g(net[i])
    # num_layers-1 is output layer
    net[num_layers - 1] = np.matmul(theta[num_layers - 1], out[num_layers - 2])
    out[num_layers - 1] = sigmoid(net[num_layers - 1])
    loss = np.sum((y-out[num_layers-1])**2)/(2*M)
    return loss, out[num_layers-1]

M = x_train.shape[0]
m = x_train.shape[1]
r = y_train.shape[1]
lr = 0.1
batch_size = 100
factor = 1e-4
if activation=='relu':
    g = relu
    d_g = d_relu
else:
    g=sigmoid
    d_g = d_sigmoid
min_lr = 5e-4

start = time.time()

theta, layers, epochs = train(M,m,hidden_layers,r,g,d_g,x_train,y_train,lr,batch_size,k,delta,adaptive)

end = time.time()

print(f"Training Time: {end-start} s")
print(f"Epochs: {epochs}")

loss, output = forward(M,layers,g,x_train,y_train,theta)
output = output.T
output = np.argmax(output,axis=1)
print(np.unique(output))
y_train2 = np.argmax(y_train,axis=1)
confusion = confusion_matrix(y_train2,output)
print("Train Data Confusion Matrix")
print(confusion)
np.savetxt(f'Q2/confusion_matrix_train_{q}_{hidden_layers[0]}_{activation}.txt',confusion)
train_acc = (output == y_train2).sum()/M

print(f"Training Accuracy: {train_acc}")

M = x_test.shape[0]

loss, output = forward(M,layers,g,x_test,y_test,theta)
output = output.T
output = np.argmax(output,axis=1)
print(np.unique(output))
y_test2 = np.argmax(y_test,axis=1)
confusion = confusion_matrix(y_test2,output)
print("Test Data Confusion Matrix")
print(confusion)
np.savetxt(f'Q2/confusion_matrix_test_{q}_{hidden_layers[0]}_{activation}.txt',confusion)
test_acc = (output==y_test2).sum()/M

print(f"Test Accuracy: {test_acc}")