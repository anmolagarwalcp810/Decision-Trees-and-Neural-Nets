import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--train',dest='train',required=False,default='bank_dataset/bank_train.csv',
                    help="Path to train file")
parser.add_argument('--test',dest='test',required=False,default='bank_dataset/bank_test.csv',
                    help="Path to test file")
parser.add_argument('--val',dest='val',required=False,default='bank_dataset/bank_val.csv',
                    help="Path to validation file")
parser.add_argument('--type',dest='two_way',required=True,default=0,type=int,help='Type of split')
args = parser.parse_args()

'arguments'
train_csv = args.train
test_csv = args.test
val_csv = args.val
two_way = args.two_way
'end'

train_data = pd.read_csv(train_csv,delimiter=';')
test_data = pd.read_csv(test_csv,delimiter=';')
val_data = pd.read_csv(val_csv,delimiter=';')

train_size = len(train_data)
test_size = len(test_data)
val_size = len(val_data)

print(train_data)

class Node:
    def __init__(self):
        self.attribute = -1
        self.children = {}
        self.is_leaf = False
        self.y = 0
        self.numerical = False
        self.median = 0

class Tree:
    def __init__(self):
        self.root = Node()
        self.count = 0
        self.count2 = 0

    def split_attribute(self,df,index):
        data_type = df.dtypes[attributes[index]]
        if data_type == 'object':
            values = df[attributes[index]].unique()
        else:
            values = [df[attributes[index]].median()]
        return values, (data_type!='object')

    def entropy(self,df):
        size = len(df)
        if (size == 0): return 0
        p0 = len(df[df['y'] == 'no']) / size
        p1 = len(df[df['y'] == 'yes']) / size
        result = 0
        if p0 > 0:
            result += p0 * np.log(p0)
        if p1 > 0:
            result += p1 * np.log(p1)
        result = -1.0 * result
        return result

    def categories(self,df,index):
        size = len(df)
        total = 0
        categories = df[attributes[index]].unique()
        for i in categories:
            n = len(df[df[attributes[index]] == i])
            p = n / size
            total += p * self.entropy(df[df[attributes[index]] == i])
        return total

    def mi(self,df,index):
        h_y = self.entropy(df)
        size = len(df)
        total = 0
        if df.dtypes[attributes[index]] == 'object':
            total = self.categories(df,index)
        else:
            median = df[attributes[index]].median()
            n1 = len(df[df[attributes[index]] <= median]) / size
            n2 = 1 - n1
            total = n1 * self.entropy(df[df[attributes[index]] <= median]) + n2 * self.entropy(
                df[df[attributes[index]] > median])
        total = h_y - total
        return total

    def choose_attribute(self,df):
        m = 0
        index = 0
        for i in range(len(attributes)):
            temp = self.mi(df, i)
            if m < temp:
                m = temp
                index = i
        return index

    def grow_tree_main(self,df,test_df,val_df):
        self.root = self.grow_tree(df,test_df,val_df)

    def grow_tree(self,df,test_df,val_df):
        self.count += 1
        new_node = Node()

        # majority
        y0 = len(df[df['y']=="no"])
        y1 = len(df[df['y']=="yes"])
        if y0>y1: new_node.y = "no"
        else: new_node.y = "yes"

        # set values
        predict_train.loc[df.index,:] = new_node.y
        predict_test.loc[test_df.index,:] = new_node.y
        predict_val.loc[val_df.index,:] = new_node.y

        # predict
        train_acc.append(pd.DataFrame(train_data['y'] == predict_train['predict'], columns=['correct']).sum()/train_size)
        test_acc.append(pd.DataFrame(test_data['y'] == predict_test['predict'], columns=['correct']).sum()/test_size)
        val_acc.append(pd.DataFrame(val_data['y'] == predict_val['predict'], columns=['correct']).sum()/val_size)

        if self.count % 1000 == 0:
            print(self.count)

        if y1==0 or y0==0:
            new_node.is_leaf = True
        else:
            attribute_index = self.choose_attribute(df)
            attribute_values, type_attribute = self.split_attribute(df, attribute_index)
            new_node.attribute = attribute_index
            new_node.numerical = type_attribute
            if type_attribute:
                # split on numerical value
                new_node.median = attribute_values[0]
                new_node.children[0] = self.grow_tree(df[df[attributes[attribute_index]] <= new_node.median],
                test_df[test_df[attributes[attribute_index]]<=new_node.median],
                val_df[val_df[attributes[attribute_index]]<=new_node.median])
                new_node.children[1] = self.grow_tree(df[df[attributes[attribute_index]] > new_node.median],
                test_df[test_df[attributes[attribute_index]]>new_node.median],
                val_df[val_df[attributes[attribute_index]]>new_node.median])
            else:
                # split on categories
                for i in attribute_values:
                    new_node.children[i] = self.grow_tree(df[df[attributes[attribute_index]] == i],
                    test_df[test_df[attributes[attribute_index]]==i],
                    val_df[val_df[attributes[attribute_index]]==i])

        return new_node

    def classify(self,df,new_df,node):
        if node.is_leaf:
            new_df.loc[df.index,:]=node.y
        else:
            if node.numerical:
                median = node.median
                self.classify(df[df[attributes[node.attribute]]<=median],new_df,node.children[0])
                self.classify(df[df[attributes[node.attribute]]>median],new_df,node.children[1])
            else:
                for i in node.children:
                    self.classify(df[df[attributes[node.attribute]]==i],new_df,node.children[i])

    def predict(self,df):
        prediction = ["" for _ in range(len(df))]
        new_df = pd.DataFrame(prediction, columns=['predict'])
        self.classify(df, new_df, self.root)
        prediction = pd.DataFrame(df['y'] == new_df['predict'], columns=['correct'])
        return prediction['correct'].sum() / len(df)

    def prune(self,node,df,val_df,test_df,node_values,prev_acc):
        count = 0
        if node.is_leaf:
            return True, 1
        else:
            flag = True
            if node.numerical:
                median = node.median
                flag1, count1 = self.prune(node.children[0], df[df[attributes[node.attribute]]<=median],
                       val_df[val_df[attributes[node.attribute]]<=median],
                        test_df[test_df[attributes[node.attribute]]<=median],
                                  node_values,prev_acc)
                flag = flag1 and flag
                count+=count1
                flag1, count1 = self.prune(node.children[1], df[df[attributes[node.attribute]]>median],
                       val_df[val_df[attributes[node.attribute]]>median],
                        test_df[test_df[attributes[node.attribute]]>median],
                                    node_values, prev_acc)
                flag = flag1 and flag
                count+=count1
            else:
                for i in node.children:
                    flag1, count1 = self.prune(node.children[i], df[df[attributes[node.attribute]]==i],
                       val_df[val_df[attributes[node.attribute]]==i],
                        test_df[test_df[attributes[node.attribute]]==i],
                                    node_values, prev_acc)
                    flag = flag1 and flag
                    count+=count1

            if flag==False:
                return False, count

            if node!=self.root:
                # store
                previous_prediction = predict_val.loc[val_df.index,:]
                # change
                predict_val.loc[val_df.index,:] = node.y
                # predict
                pruned_acc = pd.DataFrame(val_data['y'] == predict_val['predict'], columns=['correct']).sum()/val_size
                pruned_acc = pruned_acc['correct']
                # reverse changes
                predict_val.loc[val_df.index,:] = previous_prediction
                # compare
                # print(pruned_acc)
                if pruned_acc>=prev_acc:
                    if pruned_acc>node_values[0]:
                        node_values[0]=pruned_acc
                        node_values[1]=node
                        node_values[2]['train']=df.index
                        node_values[2]['val']=val_df.index
                        node_values[2]['test']=test_df.index
                        node_values[3] = count
                    return True, count
                else:
                    return False, count
            else:
                return False, count

    def prune_main(self,train_df,val_df,test_df):
        count = 0
        iterations = 0
        prev_acc = pd.DataFrame(val_data['y']==predict_val['predict'],columns=['correct']).sum()/val_size
        print(prev_acc)
        prev_acc = prev_acc['correct']
        print(prev_acc)
        while True:
            node_values = [-1,None, {'train':None,'val':None,'test':None},0]
            self.prune(self.root,train_df,val_df,test_df,node_values,prev_acc)
            if node_values[0]==-1:
                break
            node_values[1].is_leaf = True
            predict_train.loc[node_values[2]['train'],:]=node_values[1].y
            predict_val.loc[node_values[2]['val'],:]=node_values[1].y
            predict_test.loc[node_values[2]['test'],:]=node_values[1].y

            train_acc.append(pd.DataFrame(train_data['y']==predict_train['predict'],columns=['correct']).sum()/train_size)
            cur_acc = pd.DataFrame(val_data['y']==predict_val['predict'],columns=['correct']).sum()/val_size
            val_acc.append(cur_acc)
            test_acc.append(pd.DataFrame(test_data['y']==predict_test['predict'],columns=['correct']).sum()/test_size)

            count+=node_values[3]

            nodes_pruned.append(count)

            cur_acc = cur_acc['correct']

            if (cur_acc-prev_acc)<1e-8:
                break

            prev_acc = cur_acc

            if iterations%10==0:
                print(f"Nodes pruned: {count}, accuracy: {prev_acc}, iterations: {iterations}")
            iterations += 1

# two way
if two_way:
    print("Two Way Split")
    x_train = train_data.loc[:,train_data.columns!='y']
    x_test = test_data.loc[:,test_data.columns!='y']
    x_val = val_data.loc[:,val_data.columns!='y']
    x_train = pd.get_dummies(x_train)
    x_test = pd.get_dummies(x_test)
    x_val = pd.get_dummies(x_val)

    y_train = train_data.loc[:,train_data.columns=='y']
    y_test = test_data.loc[:,test_data.columns=='y']
    y_val = val_data.loc[:,val_data.columns=='y']

    train_data = x_train
    train_data['y'] = y_train
    test_data = x_test
    test_data['y'] = y_test
    val_data = x_val
    val_data['y'] = y_val

    print(train_data)

columns = train_data.columns
attributes = columns[:-1]

predict_train_arr = ["" for _ in range(len(train_data))]
predict_test_arr = ["" for _ in range(len(test_data))]
predict_val_arr = ["" for _ in range(len(val_data))]
predict_train = pd.DataFrame(predict_train_arr, columns=['predict'])
predict_test = pd.DataFrame(predict_test_arr, columns=['predict'])
predict_val = pd.DataFrame(predict_val_arr, columns=['predict'])
train_acc = []
test_acc = []
val_acc = []

tree = Tree()
tree.grow_tree_main(df=train_data,test_df=test_data,val_df=val_data)
print("nodes: ",tree.count)
print("Training Acc: ",tree.predict(train_data))
print("Test Acc: ",tree.predict(test_data))
print("Val Acc: ",tree.predict(val_data))

n = len(train_acc)
assert n==tree.count
x_axis = [i for i in range(n)]
fig0 = plt.figure(0)
plt.plot(x_axis,train_acc,label='Train Acc')
plt.plot(x_axis,test_acc,label='Test Acc')
plt.plot(x_axis,val_acc,label='Val Acc')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"Q1/accuracy_{two_way}.png")
# plt.show()

train_acc = []
test_acc = []
val_acc = []
nodes_pruned = []

tree.prune_main(train_data,val_data,test_data)
n = len(train_acc)

fig1 = plt.figure(1)
plt.plot(nodes_pruned,train_acc,label='Train Acc')
plt.plot(nodes_pruned,test_acc,label='Test Acc')
plt.plot(nodes_pruned,val_acc,label='Val Acc')
plt.xlabel("Number of Nodes Pruned")
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'Q1/prune_{two_way}.png')
# plt.show()

print("Training Acc: ",tree.predict(train_data))
print("Test Acc: ",tree.predict(test_data))
print("Val Acc: ",tree.predict(val_data))