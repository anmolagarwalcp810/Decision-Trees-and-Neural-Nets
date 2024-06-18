from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix

parser = ArgumentParser()
parser.add_argument('--train',dest='train',required=False,default='poker/poker-hand-training-true.data',
                    help="Path to train file")
parser.add_argument('--test',dest='test',required=False,default='poker/poker-hand-testing.data',
                    help="Path to test file")
args = parser.parse_args()

'arguments'
train_csv = args.train
test_csv = args.test
'end'


col_names = [str(i) for i in range(11)]
train_data = pd.read_csv(train_csv,delimiter=',',header=None,names=col_names)
test_data = pd.read_csv(test_csv,delimiter=',',header=None,names=col_names)

x_data_train = train_data.loc[:,train_data.columns!="10"]
x_data_test = test_data.loc[:,test_data.columns!="10"]

x_data_train = x_data_train.astype(str)
x_data_test = x_data_test.astype(str)

# print(x_data_train)

x_data_train = pd.get_dummies(x_data_train)
x_data_test = pd.get_dummies(x_data_test)

# print(x_data_train)

y_data_train = train_data.loc[:,train_data.columns=="10"]
y_data_test = test_data.loc[:,test_data.columns=="10"]

y_data_train = y_data_train.astype(str)
y_data_test = y_data_test.astype(str)

y_data_train = pd.get_dummies(y_data_train)
y_data_test = pd.get_dummies(y_data_test)

# print(y_data_train)

x_train = x_data_train.to_numpy()
x_test = x_data_test.to_numpy()

y_train = y_data_train.to_numpy()
y_test = y_data_test.to_numpy()

nn = MLPClassifier(hidden_layer_sizes=(100,100),activation='relu',solver='sgd',learning_rate='adaptive',
                   learning_rate_init=0.1)

nn.fit(x_train,y_train)

print(f"Train Acc: {nn.score(x_train,y_train)}\nTest Acc: {nn.score(x_test,y_test)}")

output = nn.predict(x_train)
output = np.argmax(output,axis=1)
# print(np.unique(output))
y_train2 = np.argmax(y_train,axis=1)
confusion = confusion_matrix(y_train2,output)
print("Train Data Confusion Matrix")
print(confusion)
np.savetxt(f'Q2/confusion_matrix_train_f.txt',confusion)

output = nn.predict(x_test)
output = np.argmax(output,axis=1)
# print(output)
# print(np.unique(output))
y_test2 = np.argmax(y_test,axis=1)
confusion = confusion_matrix(y_test2,output)
print("Test Data Confusion Matrix")
print(confusion)
np.savetxt(f'Q2/confusion_matrix_test_f.txt',confusion)