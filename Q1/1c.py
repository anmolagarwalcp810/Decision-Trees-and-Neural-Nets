import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--train',dest='train',required=False,default='bank_dataset/bank_train.csv',
                    help="Path to train file")
parser.add_argument('--test',dest='test',required=False,default='bank_dataset/bank_test.csv',
                    help="Path to test file")
parser.add_argument('--val',dest='val',required=False,default='bank_dataset/bank_val.csv',
                    help="Path to validation file")
args = parser.parse_args()

'arguments'
train_csv = args.train
test_csv = args.test
val_csv = args.val
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

clf = RandomForestClassifier(n_estimators=450,max_features=0.2,min_samples_split=10,bootstrap=True,oob_score=True)
print("Learning")

y_train_list = y_train['y'].tolist()
y_test_list = y_test['y'].tolist()
y_val_list = y_val['y'].tolist()

x_train_array = x_train.to_numpy()
x_test_array = x_test.to_numpy()
x_val_array = x_val.to_numpy()


# clf.fit(x_train_array,y_train_list)
# print(clf.score(x_train_array,y_train_list))
# print(clf.score(x_val_array,y_val_list))
# print(clf.score(x_test_array,y_test_list))

# print(clf.oob_score_)

parameters_grid = {"n_estimators":[50,150,250,350,450], "max_features":[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], "min_samples_split":[2,4,6,8,10]}

# parameters_grid = {"n_estimators":[275,300,325,350,375,400,425], "max_features":[0.3], "min_samples_split":[8]}

grid = ParameterGrid(param_grid=parameters_grid)
best_oob = 0
best_params = None
count = 0
print("Parameters: ",len(list(grid)))
for params in grid:
    clf = RandomForestClassifier(n_estimators=params['n_estimators'],max_features=params['max_features'],
                                 min_samples_split=params['min_samples_split'],bootstrap=True,oob_score=True)
    clf.fit(x_train_array,y_train_list)
    # if count%10==0:
    #     print(count)
    # count+=1

    if best_oob < clf.oob_score_:
        best_params = params
        best_oob = clf.oob_score_

print(f"Best parameters: \n n_estimators: {best_params['n_estimators']}\nmax_features: {best_params['max_features']}\n"
      f"min_samples_split: {best_params['min_samples_split']}")
clf = RandomForestClassifier(n_estimators=best_params['n_estimators'],max_features=best_params['max_features'],
                                 min_samples_split=best_params['min_samples_split'],bootstrap=True,oob_score=True)
clf.fit(x_train_array,y_train_list)
print("OOB Score: ",clf.oob_score_)
print("Train Acc: ",clf.score(x_train_array,y_train_list))
print("Validation Acc: ",clf.score(x_val_array,y_val_list))
print("Test Acc: ",clf.score(x_test_array,y_test_list))