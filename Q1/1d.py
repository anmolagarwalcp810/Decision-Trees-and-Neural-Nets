import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--train',dest='train',required=False,default='bank_dataset/bank_train.csv',
                    help="Path to train file")
parser.add_argument('--test',dest='test',required=False,default='bank_dataset/bank_test.csv',
                    help="Path to test file")
parser.add_argument('--val',dest='val',required=False,default='bank_dataset/bank_val.csv',
                    help="Path to validation file")
parser.add_argument('--type',dest='parameter_type',required=True,default="n_estimators",help='Type of split')
args = parser.parse_args()

'arguments'
train_csv = args.train
test_csv = args.test
val_csv = args.val
parameter_type = args.parameter_type
'end'

train_data = pd.read_csv(train_csv,delimiter=';')
test_data = pd.read_csv(test_csv,delimiter=';')
val_data = pd.read_csv(val_csv,delimiter=';')

x_train = train_data.loc[:,train_data.columns!='y']
x_test = test_data.loc[:,test_data.columns!='y']
x_val = val_data.loc[:,val_data.columns!='y']

y_train = train_data.loc[:,train_data.columns=='y']
y_test = test_data.loc[:,test_data.columns=='y']
y_val = val_data.loc[:,val_data.columns=='y']

x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)
x_val = pd.get_dummies(x_val)

y_train_list = y_train['y'].tolist()
y_test_list = y_test['y'].tolist()
y_val_list = y_val['y'].tolist()

x_train_array = x_train.to_numpy()
x_test_array = x_test.to_numpy()
x_val_array = x_val.to_numpy()

if parameter_type=='n_estimators':
    parameters_grid = {"n_estimators": [275,300,325,350,375,400,425], "max_features": [0.3], "min_samples_split": [8]}
elif parameter_type=='max_features':
    parameters_grid = {"n_estimators": [350], "max_features": [0.15,0.2,0.25,0.3,0.35,0.4,0.45],"min_samples_split": [8]}
else:
    parameters_grid = {"n_estimators":[350], "max_features":[0.3], "min_samples_split":[6,7,8,9,10]}

grid = ParameterGrid(param_grid=parameters_grid)
best_oob = 0
best_params = None
count = 0
print("Parameters: ",len(list(grid)))
train_acc = []
val_acc = []
test_acc = []
parameter_value = []
for params in grid:
    print(f'{params["n_estimators"]}, {params["max_features"]}, {params["min_samples_split"]}')
    clf = RandomForestClassifier(n_estimators=params['n_estimators'],max_features=params['max_features'],
                                 min_samples_split=params['min_samples_split'],bootstrap=True,oob_score=True)
    clf.fit(x_train_array,y_train_list)
    train_acc.append(clf.score(x_train_array, y_train_list))
    val_acc.append(clf.score(x_val_array, y_val_list))
    test_acc.append(clf.score(x_test_array, y_test_list))
    parameter_value.append(params[parameter_type])

fig0 = plt.figure(0)
plt.plot(parameter_value,train_acc,label='Train Acc')
plt.plot(parameter_value,val_acc,label='Validation Acc')
plt.plot(parameter_value,test_acc,label='Test Acc')
plt.xlabel(f'{parameter_type}')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'Q1/{parameter_type}.png')
# plt.show()

print(parameter_value)
print(train_acc)
print(val_acc)
print(test_acc)