#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from math import log
from collections import Counter


# Classification Datasets:
# * Breast Cancer
# * Car Evaluation
# * Image Segmentation
# 
# Regression Datasets:
# * Abalone
# * Computer Hardware
# * Forest Fires

# Breast Cancer Dataset:
# 
# •	Missing values handling: separate the observations with missing values based on their classes and replace the missing values with the average of the attribute in the class

# #### Breast Cancer Dataset

# In[2]:


breast_df = pd.read_csv('breast-cancer-wisconsin.data', sep = ',', header = None)


# In[3]:


breast_df.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
             'Mitoses', 'class']


# In[4]:


def fillnan(df):
    """
    this function fill in the nan values with the average values for the attribute in the class
    """
    
    df = df.replace('?', np.nan)
    
    for c in df['class'].unique():
        df[df['class'] == c] = df[df['class'] == c].apply(lambda x: x.fillna(x.dropna().astype(int).mean()), axis = 0)
        
    return df


# In[5]:


breast = fillnan(breast_df)


# In[6]:


breast = breast.drop('Sample code number', axis = 1)


# In[7]:


breast


# #### Car Dataset

# In[8]:


car = pd.read_csv('car.data', sep = ',', header = None)


# In[9]:


car.head()


# In[10]:


with open('car.names') as carname:
    for f in carname:
        print(f)


# In[11]:


car.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


# #### Segentation Dataset

# In[12]:


seg_data = []
with open('segmentation.data') as segmentation:
    for f in segmentation:
        seg_data.append(f)
seg_df = pd.DataFrame(seg_data)


# In[13]:


seg_final_df = seg_df[5:][0].str.split(',', expand = True)


# In[14]:


seg_list = ['class']
for i in seg_df.loc[3].str.split(','):
    seg_list.extend(i)


# In[15]:


seg_final_df.columns = seg_list


# In[16]:


seg_final_df = seg_final_df.rename(columns = {'HUE-MEAN\n': 'HUE-MEAN'})


# In[17]:


seg_final_df['HUE-MEAN'] = seg_final_df['HUE-MEAN'].str.rstrip()


# In[18]:


seg_final_df = seg_final_df.reset_index(drop = True)


# In[19]:


seg_final_df['class'] = pd.factorize(seg_final_df['class'])[0]


# In[20]:


segmentation = seg_final_df.drop(['REGION-PIXEL-COUNT'], axis = 1)


# In[21]:


segmentation


# In[22]:


segmentation.loc[:, segmentation.columns != 'class'] = segmentation.loc[:, segmentation.columns != 'class'].astype(float)


# In[23]:


segmentation


# – Provide sample outputs from one test set on one fold for a classification tree and a regression tree.
# 
# – Show a sample classification tree without pruning and with pruning as well as a sample regression
# tree without early stopping and with early stopping.
# 
# – Demonstrate the calculation of information gain, gain ratio, and mean squared error.
# 
# – Demonstrate a decision being made to prune a subtree (pruning) and a decision being made to
# stop growing a subtree (early stopping).
# 
# – Demonstrate an example traversing a classification tree and a class label being assigned at the
# leaf.
# 
# – Demonstrate an example traversing a regression tree and a prediction being made at the leaf.
# 
# – Show the average performance over the five folds on a classification data set (with and without
# pruning) and on a regression data set (with and without early stopping).

# In[33]:


## Create training set, test set and tuning set


# In[34]:


def tune_remaining(data_df):
    """
    take out 10% of the classification dataset for tuning and 90% for training and testing
    """
    
    class_df = pd.DataFrame(data_df.groupby(['class'])['class'].count()).rename(columns = {'class': 'class_proportion'})
    tune_df = pd.DataFrame()
    index_list = []
    
    for c in class_df.index.unique():
        c_df = data_df[data_df['class'] == c].sample(n = int(np.ceil(len(data_df[data_df['class'] == c]) * 0.1)))
        index_list.append(c_df.index.tolist())
        c_df = c_df.reset_index(drop = True)
        tune_df = pd.concat([tune_df, c_df], axis = 0)
        
    tune_df = tune_df.reset_index(drop = True)
    
    flatten_list = [y for x in index_list for y in x]
    
    remaining_df = data_df[~data_df.index.isin(flatten_list)].reset_index(drop = True)
    
    return tune_df, remaining_df
        


# In[36]:


def create_fold_classification(data_df):
    """
    create 5 subsets for the classification dataset
    """
    # calculate the class proportion
    class_df = pd.DataFrame(data_df.groupby(['class'])['class'].count()).rename(columns = {'class': 'class_proportion'})
    
    # calculate the fold size
    class_df['fold_size'] = np.ceil(class_df['class_proportion']/5)
    class_df['fold_size'] = class_df['fold_size'].astype(int)
    
    # create a dictionary to store the data in each fold
    keys = [i for i in range(1, 6)]
    data_subset_dict = {k: [] for k in keys}
    
    for c in class_df.index.unique():
        class_subset = shuffle(data_df[data_df['class'] == c]).reset_index(drop = True)

        # get the fold size
        fold_size = class_df.loc[c, 'fold_size']

        # append the subsets into a dictionary
        j = 0
        while j <= len(class_subset):
            k = 1
            while k <= len(data_subset_dict):
                data_subset_dict[k].append(class_subset.iloc[j:j+fold_size])
                k = k + 1
                j = j + fold_size
                if (j >= len(class_subset)) or (k > len(data_subset_dict)):
                    break
                    
    return data_subset_dict


# In[38]:


def train_test_c(data_subset_dict, num):
    
    """
    build a training set with 80% of the data and test set with 20% of the data for classification dataset
    """
    
    # get the dictionary key for the training set
    range_list = [x for x in range(1, len(data_subset_dict)+1) if x != num]
    
    # build the training set
    training_set = pd.DataFrame()
    for i in range_list:
        for j in range(0, len(data_subset_dict[i])):
            training_set = pd.concat([training_set, (data_subset_dict[i][j])])
    
    training_set = training_set.reset_index(drop = True)
    
    test_set = pd.DataFrame()
    for k in range(0, len(data_subset_dict[num])):
        test_set = pd.concat([test_set, data_subset_dict[num][k]])
    
    test_set = test_set.reset_index(drop = True)
    return training_set, test_set


# In[40]:


class TreeNode:
    def __init__(self, class_label):
        # current attribute's class label
        self.class_label = class_label
        # current attribute's parent
        self.parent = None
        # current attribute's parent's unique values 
        self.parent_val = []
        # current attribute
        self.attribute = None
        # current attribute's unique value
        self.attribute_val = []
        # subsets of the data instances where the attribute value = one of the unique values
        self.data_instances = []
        # a dictionary that stores the attribute's child unique attribute value and the subtree
        self.child = {}
        self.prune = False


# In[41]:


def entropy_numeric(data, attribute):
    """
    calculate the entropy for the numerical attributes
    """
    
    a_dict = data.groupby(['class'])[attribute].count().to_dict()
    
    item_list = []
    for items in a_dict.values():
        item_list.append(items / sum(a_dict.values()) * np.log2(items / sum(a_dict.values())))
        
    return -np.sum(item_list)
    


# In[42]:


def info_gain_numeric(df, attribute, left_node, right_node, entropy_before, split_point):
    
    """
    calculate the information gain for the numeric attribute
    """
    n = len(df)
    
    entropy_l = entropy_numeric(left_node, attribute)

    entropy_r = entropy_numeric(right_node, attribute)
    
    t_entropy = (len(left_node) / n) * entropy_l + (len(right_node) / n) * entropy_r
    
    info_gain = entropy_before - t_entropy
    
    return info_gain
    


# In[43]:


def optimal_split(data, attribute):
    
    """
    find the optimal split point
    """
    
    entropy_before = prior_entropy(data)
    
    info_gains = {}
    
    binary_point_list = [(data.loc[i, attribute] + data.loc[i-1, attribute]) / 2 for i in range(1, len(data)) if data.loc[i, 'class'] != data.loc[i - 1, 'class']]
    
    for val in binary_point_list:
        
        left_node = data.loc[data[attribute] < val]
        
        right_node = data.loc[data[attribute] > val]

        info_gains[val] = info_gain_numeric(data, attribute, left_node, right_node, entropy_before, val)
        
    attribute_split = list(info_gains.keys())[list(info_gains.values()).index(max(info_gains.values()))]
    
    return attribute_split, max(info_gains.values())


# In[44]:


def attribute_selection_numeric(data_df):
    """
    select attribute based on the information gain
    """
    info_gain = {}

    for attribute in data_df.loc[:, data_df.columns != 'class'].columns:
        data = data_df[[attribute, 'class']].sort_values([attribute, 'class']).reset_index(drop = True)
        info_gain[attribute, optimal_split(data, attribute)[0]] = optimal_split(data, attribute)[1]
    
    attribute = list(info_gain.keys())[list(info_gain.values()).index(max(info_gain.values()))][0]
    split_val = list(info_gain.keys())[list(info_gain.values()).index(max(info_gain.values()))][1]
    
    return attribute, split_val


# In[45]:


def most_common(data_df):
    """
    Return the most common class
    """
    return data_df.groupby(['class'])['class'].count().rename(columns = {'class': 'class_count'}).reset_index().max()['class']


# In[46]:


def prior_entropy(data_df):
    """
    Calculate the prior entropy
    """
    if len(data_df['class'].unique()) == 1:
        entropy = 0
    else:
        # calculate the prior entropy value
        prior_df = pd.DataFrame(data_df.groupby(['class'])['class'].count() / len(data_df['class'])).rename(columns = {'class' :'probability'}) 
        prior_df['entropy'] = np.log2(prior_df['probability']) * prior_df['probability']
        entropy = -prior_df['entropy'].sum()
    return entropy


# In[47]:


def entropy(data_df, attribute):
    """
    calculate the information gain for each attribute
    """
    
    if len(data_df[[attribute, 'class']]['class'].unique()) == 1:
        return 0

    else:
        # calculate the entropy value
        attribute_class = pd.DataFrame(data_df.groupby([attribute, 'class'])['class'].count()).rename(columns = {'class': 'attribute_count'}).reset_index()
        attribute_df = pd.DataFrame(data_df.groupby([attribute])[attribute].count()).rename(columns = {attribute : 'total_count'}).reset_index()
        a_df = pd.merge(attribute_class, attribute_df, on = attribute)
        a_df['prob'] = a_df['attribute_count'] / a_df['total_count']
        a_df['I_prob'] = a_df['prob'] * np.log2(a_df['prob'])
        # create counter to store the value
        c_iprob = Counter(a_df.groupby([attribute])['I_prob'].sum().to_dict())
        c_prob = Counter(data_df.groupby([attribute])[attribute].count().to_dict())

        entropy = 0

        for i, item in c_prob.items():
            entropy += (c_iprob[i] * item/sum(c_prob.values()))
    return -entropy


# In[48]:


def information_gain(data_df, attribute):
    """
    calculate the information gain ratio
    """
    gain = prior_entropy(data_df) - entropy(data_df, attribute)
    attribute_class = pd.DataFrame(data_df.groupby([attribute, 'class'])['class'].count()).rename(columns = {'class': 'attribute_count'}).reset_index()
    attribute_df = pd.DataFrame(data_df.groupby([attribute])[attribute].count()).rename(columns = {attribute : 'total_count'}).reset_index()
    a_df = pd.merge(attribute_class, attribute_df, on = attribute)
    a_df['prob'] = a_df['attribute_count'] / a_df['total_count']
    attribute_df = pd.DataFrame(data_df.groupby([attribute])[attribute].count()).rename(columns = {attribute : 'total_count'}).reset_index()
    attribute_df['prob'] = attribute_df['total_count']/attribute_df['total_count'].sum()
    attribute_df['split_prob'] = attribute_df['prob'] * np.log2(attribute_df['prob'])
    split_info = -attribute_df['split_prob'].sum()
    if split_info == 0:
        gain_ratio = -100
    else:
        gain_ratio = gain/split_info
    return gain_ratio


# In[49]:


def attribute_selection(data_df):
    """
    select attribute based on the gain ratio
    """
    gain_ratio_list = {}
    for i in data_df.iloc[:, (data_df.columns != 'class')].columns:
        gain_ratio = information_gain(data_df, i)
        gain_ratio_list[i] = gain_ratio
    return max(gain_ratio_list, key = gain_ratio_list.get)


# In[50]:


def predict(f_tree, val_set_row):
    """
    algorithm for making prediction
    """
    # if it is a leaf node, return the class label
    if len(f_tree.child) == 0:
        return f_tree.class_label
    
    else:
        # select the column values from the test set with the attribute in the tree
        a_val = val_set_row[f_tree.attribute]
        
        # assume we don't want to prune the tree, we go to the next level of the tree until we reach the leaf node
        if a_val in f_tree.child and f_tree.child[a_val].prune != True:
            return predict(f_tree.child[a_val], val_set_row)
        
        else:
            # if we want to prune the tree, return the common class from the subtree
            df = pd.DataFrame()
            for a in f_tree.attribute_val:
                df = pd.concat([df, f_tree.child[a].data_instances], axis = 0)
            return most_common(df)
            


# In[51]:


def predict_numeric(f_tree, val_set_row):
    """
    algorithm for making prediction
    """
    # if it is a leaf node, return the class label
    if len(f_tree.child) == 0:
        return f_tree.class_label
    
    else:
        # select the column values from the test set with the attribute in the tree
        a_val = val_set_row[f_tree.attribute]
        
        if a_val <= float(list(f_tree.child.keys())[0].split('_')[1]) and (f_tree.child[list(f_tree.child.keys())[0]].prune == False):
            return predict_numeric(f_tree.child[list(f_tree.child.keys())[0]], val_set_row)
        elif a_val > float(list(f_tree.child.keys())[0].split('_')[1]) and (f_tree.child[list(f_tree.child.keys())[1]].prune == False):
            return predict_numeric(f_tree.child[list(f_tree.child.keys())[1]], val_set_row)
        
        else:
            # if we want to prune the tree, return the common class from the subtree
            df = pd.DataFrame()
            
            for k in list(f_tree.child.keys()):
                df = pd.concat([df, f_tree.child[k].data_instances], axis = 0)
            return most_common(df)


# In[52]:


def pruning(TREE, f_tree, tune_df, data_type):
    """
    Algorithm for pruning the tree
    """

    # if this is a leaf node
    if len(f_tree.child) == 0:
        error_before = classification_error(TREE, tune_df, data_type)
        #print('error before tuning', error_before)
        # prune the tree
        f_tree.prune = True

        #print('error after tuning', classification_error(TREE, tune_df, data_type))

        # if the error before pruning is less than after pruning, we stop
        if error_before <= classification_error(TREE, tune_df, data_type):
            f_tree.prune = False
        return 

    branches = f_tree.child

    for attribute, subtree in branches.items():
        #print("attribute", branches[attribute].attribute)
        pruning(TREE, subtree, tune_df, data_type)

    error_before = classification_error(TREE, tune_df, data_type)
    #print('it is not leaf node, error before tuning', error_before)
    # prune the tree
    f_tree.prune = True

    # if the error before pruning is less than after pruning, we stop
    if error_before < classification_error(TREE, tune_df, data_type):
        #print('it is not leaf node, error after tuning', classification_error(TREE, tune_df, data_type))
        f_tree.prune = False


# In[53]:


def classification_error(f_tree, tune_df, data_type):
    """
    calculate the classification error for the classification tree
    """
    tune_prediction_list = []
    for i in range(len(tune_df)):
        tune_set_row = tune_df.iloc[i, :]
        if data_type == 'categorical':
            tune_predictions = predict(f_tree, tune_set_row)
        elif data_type == 'numerical':
            tune_predictions = predict_numeric(f_tree, tune_set_row)
        tune_prediction_list.append(tune_predictions)
        
    true_labels = tune_df['class'].to_list()
    
    counter = 0
    for i in range(len(true_labels)):
        if true_labels[i] != tune_prediction_list[i]:
            counter += 1
    
    c_error = counter / len(true_labels)
    return c_error


# In[54]:


def ID3_categorical(data_df, class_label):
    """
    ID3 Algorithm 
    """
    
    # if there is no more instances, the leaf node is labeled with the most common class
    if len(data_df) == 0:
        most_common_label = most_common(data_df)
        return TreeNode(most_common_label)
    
    # if every instance in the subset is in the same class, the leaf node is labeled as the class of the instances
    elif len(data_df.groupby(['class'])['class'].count()) == 1:
        class_label = data_df.groupby(['class'])['class'].count().keys()[0]
        return TreeNode(class_label)
    
    else:
        # initialize the tree
        most_common_label = most_common(data_df)
        f_tree = TreeNode(most_common_label)
        # set the root node
        attribute_selected = attribute_selection(data_df)
        f_tree.attribute = attribute_selected

        # get the attribute value for the root node
        a_selected_val = list(data_df[attribute_selection(data_df)].unique())
        f_tree.attribute_val = a_selected_val
        
        # split the instances
        for val in f_tree.attribute_val:
            subset = data_df[data_df[attribute_selected] == val].reset_index(drop = True)
            subtree = ID3_categorical(subset, most_common(subset))
            subtree.data_instances = subset
            subtree.parent =  attribute_selected
            subtree.parent_val = a_selected_val
            f_tree.child[val] = subtree
            
        return f_tree


# In[62]:


def ID3_numerical(data_df, class_label):
    """
    ID3 Algorithm 
    """
    
    # if there is no more instances, the leaf node is labeled with the most common class
    if len(data_df) == 0:
        most_common_label = most_common(data_df)
        return TreeNode(most_common_label)
    
    # if every instance in the subset is in the same class, the leaf node is labeled as the class of the instances
    elif len(data_df.groupby(['class'])['class'].count()) == 1:
        class_label = data_df.groupby(['class'])['class'].count().keys()[0]
        return TreeNode(class_label)
    
    else:
        # initialize the tree
        most_common_label = most_common(data_df)
        f_tree = TreeNode(most_common_label)
        
        # set the root node
        attribute_selected, a_split_val = attribute_selection_numeric(data_df)
        
        f_tree.attribute = attribute_selected
        
        for val in ['left_' + str(a_split_val), 'right_' + str(a_split_val)]:

            if val == 'left_' + str(a_split_val):
                subset = data_df[data_df[attribute_selected] <= a_split_val].reset_index(drop = True)

            else:
                subset = data_df[data_df[attribute_selected] > a_split_val].reset_index(drop = True)

            
            subset = subset.loc[:, subset.columns != attribute_selected]

            subtree = ID3_numerical(subset, most_common(subset))

            subtree.data_instances = subset
            subtree.parent = attribute_selected
            f_tree.child[val] = subtree
            
        return f_tree


# In[70]:


def print_branches(tree):
    
    """
    print branches of the tree
    """
    
    if len(tree.child) == 0:
        print('Subtree Complete')
        print('Class Label is ', tree.class_label)
    
    branches = tree.child
    
    for attribute, subtree in branches.items():
        print('Attribute is', branches[attribute].parent)
        print('Attribute Value is', attribute)
        print_branches(branches[attribute])


# In[76]:


def cross_validation_unprune(data_subset_dict, data_type, most_common_label):
    """
    do five fold cross validatiaon with the training and test set for unpruning tree
    """
    
    test_result_unpruned = {}
    
    for num in range(1, 6):
        
        training_set, test_set = train_test_c(data_subset_dict, num)
        
        if data_type == 'categorical':
            f_tree = ID3_categorical(training_set, most_common_label)
        elif data_type == 'numerical':
            f_tree = ID3_numerical(training_set, most_common_label)
        
        unpruned_error = classification_error(f_tree, test_set, data_type)
        
        test_result_unpruned[num] = unpruned_error
        
        test_average = sum(test_result_unpruned.values()) / len(test_result_unpruned)
        
    return test_result_unpruned, test_average
        
        


# In[77]:


def cross_validation_prune(data_subset_dict, data_type, most_common_label, tune_df):
    """
    do five fold cross validatiaon with the training and test set for pruning tree
    """
    
    test_result_pruned = {}
    
    for num in range(1, 6):
        
        training_set, test_set = train_test_c(data_subset_dict, num)
        
        if data_type == 'categorical':
            f_tree = ID3_categorical(training_set, most_common_label)
        elif data_type == 'numerical':
            f_tree = ID3_numerical(training_set, most_common_label)
            
        global TREE
        
        TREE = f_tree
        
        pruning(TREE, f_tree, tune_df, data_type)
            
        pruned_error = classification_error(TREE, test_set, data_type)
        
        test_result_pruned[num] = pruned_error
        
        test_average = sum(test_result_pruned.values()) / len(test_result_pruned)
        
    return test_result_pruned, test_average


# In[78]:


def c_prune_unprune(data_df, data_type):
    
    """
    Return accuracy for classification tree with and without pruning using cross validation
    """
    most_common_label = most_common(data_df)
    tune_df, remaining_df = tune_remaining(data_df)
    data_subset_dict = create_fold_classification(data_df)
    c_unprune = cross_validation_unprune(data_subset_dict, data_type, most_common_label)
    test_result_pruned, prune = cross_validation_prune(data_subset_dict, data_type, most_common_label, tune_df)
    test_result_unpruned, unprune = cross_validation_unprune(data_subset_dict, data_type, most_common_label)
    return test_result_pruned, prune, test_result_unpruned, unprune


# In[79]:


test_result_pruned_breast, prune_breast, test_result_unpruned_breast, unprune_breast = c_prune_unprune(breast, 'categorical')


# In[80]:


test_result_pruned_breast, prune_breast, test_result_unpruned_breast, unprune_breast


# In[81]:


test_result_pruned_car, prune_car, test_result_unpruned_car, unprune_car = c_prune_unprune(car, 'categorical')


# In[82]:


test_result_pruned_car, prune_car, test_result_unpruned_car, unprune_car


# In[83]:


test_result_pruned_seg, prune_seg, test_result_unpruned_seg, unprune_seg = c_prune_unprune(segmentation, 'numerical')


# In[84]:


test_result_pruned_seg, prune_seg, test_result_unpruned_seg, unprune_seg


# In[ ]:





# – Provide sample outputs from one test set on one fold for a classification tree

# In[ ]:


# car dataset
data_df = car
most_common_label = most_common(data_df)
tune_df, remaining_df = tune_remaining(data_df)
data_subset_dict = create_fold_classification(data_df)
num = 1
training_set, test_set = train_test_c(data_subset_dict, num)


# In[ ]:


f_tree_unpruned = ID3_categorical(training_set, most_common_label)


# In[ ]:


test_df = test_set
data_type = 'categorical'
test_prediction_list = []
for i in range(len(test_df)):
    test_set_row = test_df.iloc[i, :]
    if data_type == 'categorical':
        test_predictions = predict(f_tree_unpruned, test_set_row)
    elif data_type == 'numerical':
        test_predictions = predict_numeric(f_tree_unpruned, test_set_row)
    test_prediction_list.append(test_predictions)


# In[ ]:


test_prediction_list


# In[ ]:


true_labels = test_df['class'].to_list()


# In[ ]:


true_labels


# – Show a sample classification tree without pruning and with pruning

# In[ ]:


# without pruning
print_branches(f_tree_unpruned)


# In[ ]:


# pruned
f_tree = ID3_categorical(training_set, most_common_label)
global TREE
TREE = f_tree
pruning(TREE, f_tree, tune_df, 'categorical')
# after pruning
classification_error(f_tree, test_set, 'categorical')


# In[ ]:


# with pruning
print_branches(f_tree)


# – Demonstrate the calculation of information gain, gain ratio

# In[ ]:


df = training_set
attribute = 'safety'
gain = prior_entropy(data_df) - entropy(data_df, attribute)
print('Information Gain', gain)
attribute_class = pd.DataFrame(data_df.groupby([attribute, 'class'])['class'].count()).rename(columns = {'class': 'attribute_count'}).reset_index()
attribute_df = pd.DataFrame(data_df.groupby([attribute])[attribute].count()).rename(columns = {attribute : 'total_count'}).reset_index()
a_df = pd.merge(attribute_class, attribute_df, on = attribute)
a_df['prob'] = a_df['attribute_count'] / a_df['total_count']
attribute_df = pd.DataFrame(data_df.groupby([attribute])[attribute].count()).rename(columns = {attribute : 'total_count'}).reset_index()
attribute_df['prob'] = attribute_df['total_count']/attribute_df['total_count'].sum()
attribute_df['split_prob'] = attribute_df['prob'] * np.log2(attribute_df['prob'])
split_info = -attribute_df['split_prob'].sum()
if split_info == 0:
    gain_ratio = -100
else:
    gain_ratio = gain/split_info
    
print('Gain Ratio', gain_ratio)


# – Demonstrate a decision being made to prune a subtree (pruning)

# In[ ]:


def pruning(TREE, f_tree, tune_df, data_type):
    """
    Algorithm for pruning the tree
    """

    # if this is a leaf node
    if len(f_tree.child) == 0:
        error_before = classification_error(TREE, tune_df, data_type)
        print('error before tuning', error_before)
        # prune the tree
        f_tree.prune = True
        print('error after tuning', classification_error(TREE, tune_df, data_type))

        # if the error before pruning is less than after pruning, we stop
        if error_before < classification_error(TREE, tune_df, data_type):
            f_tree.prune = False
            print('stop pruning')
        return 

    branches = f_tree.child

    for attribute, subtree in branches.items():
        print("attribute", branches[attribute].attribute)
        pruning(TREE, subtree, tune_df, data_type)

    error_before = classification_error(TREE, tune_df, data_type)
    print('it is not leaf node, error before tuning', error_before)
    # prune the tree
    f_tree.prune = True

        # if the error before pruning is less than after pruning, we stop
    if error_before < classification_error(TREE, tune_df, data_type):
        print('it is not leaf node, error after tuning', classification_error(TREE, tune_df, data_type))
        f_tree.prune = False
        print('stop pruning')


# In[ ]:


# pruned
f_tree = ID3_categorical(training_set, most_common_label)
global TREE
TREE = f_tree
pruning(TREE, f_tree, tune_df, 'categorical')


# – Demonstrate an example traversing a classification tree and a class label being assigned at the
# leaf.

# In[ ]:


def predict(f_tree, val_set_row):
    """
    algorithm for making prediction
    """
    # if it is a leaf node, return the class label
    if len(f_tree.child) == 0:
        print('Class Label Assigned:', f_tree.class_label)
        return f_tree.class_label
    
    else:
        # select the column values from the test set with the attribute in the tree
        a_val = val_set_row[f_tree.attribute]
        
        # assume we don't want to prune the tree, we go to the next level of the tree until we reach the leaf node
        if a_val in f_tree.child and f_tree.child[a_val].prune != True:
            print('Attribute at the Node', f_tree.child[a_val].parent)
            return predict(f_tree.child[a_val], val_set_row)
        
        else:
            # if we want to prune the tree, return the common class from the subtree
            df = pd.DataFrame()
            for a in f_tree.attribute_val:
                df = pd.concat([df, f_tree.child[a].data_instances], axis = 0)
            return most_common(df)


# In[ ]:


i = 0
val_set_row = test_set.iloc[i, :]
predict(f_tree_unpruned, val_set_row)

test_set.iloc[i, :]

# – Show the average performance over the five folds on a classification data set (with and without
# pruning) 

# In[ ]:


test_result_pruned_car, prune_car, test_result_unpruned_car, unprune_car = c_prune_unprune(car, 'categorical')


# In[ ]:


test_result_pruned_car, prune_car, test_result_unpruned_car, unprune_car




# Regression Datasets:
# * Abalone
# * Computer Hardware
# * Forest Fires

# #### Abalone Dataset

# In[2]:


abalone = pd.read_csv('abalone.data', sep = ',', header = None)


# In[3]:


abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'target']


# In[4]:


abalone.head()


# #### Computer Hardware Dataset

# In[5]:


machine = pd.read_csv('machine.data', header = None)


# In[6]:


machine.columns = ['Vendor name', 'Model Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'target', 'ERP']


# In[7]:


machine = machine.drop(['Model Name', 'ERP'], axis = 1)


# In[8]:


machine.head()


# #### Forestfire Dataset

# In[9]:


forestfires = pd.read_csv('forestfires.data')


# In[10]:


forestfires = forestfires.rename(columns = {'area': 'target'})


# – Provide sample outputs from one test set on one fold for a classification tree and a regression tree.
# 
# – Show a sample classification tree without pruning and with pruning as well as a sample regression
# tree without early stopping and with early stopping.
# 
# – Demonstrate the calculation of information gain, gain ratio, and mean squared error.
# 
# – Demonstrate a decision being made to prune a subtree (pruning) and a decision being made to
# stop growing a subtree (early stopping).
# 
# – Demonstrate an example traversing a classification tree and a class label being assigned at the
# leaf.
# 
# – Demonstrate an example traversing a regression tree and a prediction being made at the leaf.
# 
# – Show the average performance over the five folds on a classification data set (with and without
# pruning) and on a regression data set (with and without early stopping).

# In[11]:


## Create training set, test set and tuning set


# In[12]:


def train_tune_test(df):
    
    """
    take out 10% of the regression dataset for tuning and 90% for training and testing
    """
    
    # get the tuning set
    X_tune = df.sample(frac = 0.1)
    
    # get the remaining set
    X_remaining = df[~df.index.isin(X_tune.index)]
    
    X_tune = X_tune.reset_index(drop = True)
    
    X_remaining = X_remaining.reset_index(drop = True)
    
    return X_tune, X_remaining


# In[13]:


def create_fold_regression(data_df):
    
    """
    create 5 subsets for the regression dataset
    """
    
    # shuffle the dataset
    shuffle_df = shuffle(data_df).reset_index(drop = True)
    
    # calculate the fold size
    fold_size = int(np.floor(len(shuffle_df) / 5))
    
    
    # create a dictionary to store the data in each fold
    keys = [i for i in range(1, 6)]
    data_subset_dict = {k: [] for k in keys}
    
    j = 0
    k = 1
    while j <= len(shuffle_df):
        while k <= len(data_subset_dict):
            data_subset_dict[k] = shuffle_df.iloc[j: j + fold_size]
            k = k + 1
            j = j + fold_size
        if (j >= len(shuffle_df)) or (k >= len(data_subset_dict)):
            break
            
    return data_subset_dict


# In[14]:


def train_test_r(data_subset_dict, num):
    """
    build a training set with 80% of the data and test set with 20% of the data for regression dataset
    """
    
    # get the dictionary key for the training set
    range_list = [x for x in range(1, len(data_subset_dict)+1) if x != num]

    # build the training set
    training_set = pd.DataFrame()
    for i in range_list:
        training_set = pd.concat([training_set, (data_subset_dict[i])])

    training_set = training_set.reset_index(drop = True)

    test_set = pd.DataFrame()
    test_set = pd.concat([test_set, data_subset_dict[num]])

    test_set = test_set.reset_index(drop = True)
    
    return training_set, test_set


# In[15]:


class TreeNode:
    def __init__(self, target):
        self.attribute = None
        self.target = target
        self.data_instances = None
        self.left = []
        self.right = []
        self.parent = None
        self.child = {}
        


# In[16]:


def split_point(df, col, n):
    """
    calculate the split point for the numerical attribute
    """

    col_unique = np.unique(df.loc[:, col])
    
    split_points = [col_unique[int(len(col_unique) / n) * (i + 1)] for i in range(n-1)]
    
    return split_points


# In[17]:


def mse(subset):
    """
    calculate the mean square error 
    """
    return np.mean((subset - np.mean(subset))**2)


# In[18]:


def tree_build(data_df, n):
    
    """
    algorithm for building regression tree
    """
    
    # get all the attributes 
    attribute_list = data_df.loc[:, data_df.columns != 'target'].columns
    
    #print('attribute_list', attribute_list)

    # get the split points for the numeric attributes
    a_dict = {}
    
    for col in attribute_list:
        if data_df[col].dtypes != 'object':
            data_df[col] = data_df[col].astype(float)
            a_dict[col] = split_point(data_df, col, n)
    
    #print(a_dict)
    # get the attribute that has the most information gain
    attribute_s = None
    attribute_threshold = None
    attribute_mse = 100000000000000000
    
    
    for col in attribute_list:

        if data_df[col].dtypes == 'object':
            total_mse = 0
            for d in data_df[col].unique():
                sub_df = data_df[data_df[col] == d]['target']
                len_df = len(sub_df)
                total_mse += (len(sub_df) / len(data_df)) * mse(sub_df)
                
                if total_mse < attribute_mse:
                    attribute_mse = total_mse
                    attribute_s = col
                    attribute_threshold = d
                
                
    #print('best information/attribute/threshold', attribute_gain, attribute_s, attribute_threshold)
    #print('total mse', attribute_mse)

    for a in a_dict.keys():
        #print('attribute selected:', a)
        thresholds = a_dict[a]
        
        # iterate each threshold
        for t in thresholds:
            #print('thresholds: ', t)
            #print('attribute for this threshold is', a)
            left_t = data_df[data_df[a] <= t]['target']
            #print('left_df', left_t)
            left_n = len(data_df[data_df[a] <= t])
            #print('left_df', left_n)
            mse_l = mse(left_t)
            #print('mse left', mse_l)

            right_t = data_df[data_df[a] > t]['target']
            right_n = len(data_df[data_df[a] > t])
            #print('right_df', right_n)
            mse_r = mse(right_t)
            #print('mse right', mse_r)
            
            # information gain
            total_mse = ((left_n / len(data_df)) * mse_l + ((right_n) / len(data_df)) * mse_r)
            
            #print('total mse', total_mse)
            
            if total_mse < attribute_mse:
                attribute_mse = total_mse
                attribute_s = a
                attribute_threshold = t
                #print('best information/attribute/threshold', gain, a, t)  
    

    return attribute_mse, attribute_s, attribute_threshold


# In[19]:


def split_df(data_df, attribute, threshold):
    
    #print('threshold', threshold)
    
    """
    split the dataframe based on the selected threshold
    """
    #print('data_df_shape', data_df.shape)
    
    if data_df[attribute].dtypes == 'object':
        data_df_left = data_df[data_df[attribute] == threshold].reset_index(drop = True)
        data_df_right = data_df[data_df[attribute] != threshold].reset_index(drop = True)
    
    else:    
        data_df_left = data_df[data_df[attribute] <= threshold].reset_index(drop = True)
        data_df_right = data_df[data_df[attribute] > threshold].reset_index(drop = True)
    
    #print('data_left', data_df_left.shape)
    
    return data_df_left, data_df_right
    


# In[20]:


def CART(data_df, n):
    """
    Cart Algorithm
    """
    if len(data_df['target'].unique()) == 1:
        target_val = data_df['target'].unique()[0]
        return TreeNode(target_val)
    
    
    else:
        target_val = np.mean(data_df['target'])
        f_tree = TreeNode(target_val)
        attribute_mse, attribute_selected, threshold_selected = tree_build(data_df, n)
        #print('attribute selected', attribute_selected)
        
        if attribute_selected == None:
            target_val = np.mean(data_df['target'])
            return TreeNode(target_val)
        else:
            #print('attribute', attribute_selected)
            f_tree.attribute = attribute_selected
            data_df_left, data_df_right = split_df(data_df, attribute_selected, threshold_selected)
            for val in ['left_' + str(threshold_selected), 'right_' + str(threshold_selected)]:
                if val == 'left_' + str(threshold_selected):
                    subset = data_df_left.loc[:, data_df_left.columns != attribute_selected]
                else:
                    subset = data_df_right.loc[:, data_df_right.columns != attribute_selected]
                
                subtree = CART(subset, n)
                subtree.parent = attribute_selected
                f_tree.child[val] = subtree
        
    return f_tree
    


# In[21]:


def predict_numeric(data_df, f_tree, val_set_row):
    """
    algorithm for making prediction
    """
    # if it is a leaf node, return the class label
    if len(f_tree.child) == 0:
        return f_tree.target
    
    else:
        # select the column values from the test set with the attribute in the tree
        a_val = val_set_row[f_tree.attribute]
        
        if data_df[f_tree.attribute].dtypes == 'object':
        
            if a_val == list(f_tree.child.keys())[0].split('_')[1]:
                return predict_numeric(data_df, f_tree.child[list(f_tree.child.keys())[0]], val_set_row)
            elif a_val != list(f_tree.child.keys())[0].split('_')[1]:
                return predict_numeric(data_df, f_tree.child[list(f_tree.child.keys())[1]], val_set_row)
                
        else:
            if a_val <= float(list(f_tree.child.keys())[0].split('_')[1]):
                return predict_numeric(data_df, f_tree.child[list(f_tree.child.keys())[0]], val_set_row)
            elif a_val > float(list(f_tree.child.keys())[0].split('_')[1]):
                return predict_numeric(data_df, f_tree.child[list(f_tree.child.keys())[1]], val_set_row)
        


# In[22]:


def regression_error(f_tree, tune_df):
    """
    calculate the regression error for the regression tree
    """
    
    tune_prediction_list = []
    for i in range(len(tune_df)):
        tune_set_row = tune_df.iloc[i, :]
        tune_predictions = predict_numeric(tune_df, f_tree, tune_set_row)
        tune_prediction_list.append(tune_predictions)
        
    true_target = tune_df['target'].to_list()
    
    diff_list = []
    for i in range(len(true_target)):
        diff_list.append((true_target[i] - tune_prediction_list[i])**2)
        
    
    return np.mean(diff_list)


# In[23]:


def print_branches(tree):
    
    """
    print branches of the tree
    """
    
    if len(tree.child) == 0:
        print('Subtree Complete')
        print('Class Label is ', tree.target)
    
    branches = tree.child
    
    for attribute, subtree in branches.items():
        print('Attribute is', branches[attribute].parent)
        print('Attribute Value is', attribute)
        print_branches(branches[attribute])


# In[24]:


def tune_bins(data_df):
    """
    do five fold cross validatiaon with the training and test set for unpruning tree
    """
    
    bins_dict = {}
    
    tune_df, remaining_df = train_tune_test(data_df)
    data_subset_dict = create_fold_regression(data_df)
    training_set, test_set = train_test_r(data_subset_dict, num = 1)
    
    # Split points
    for n in [4, 6, 10]:
        
        f_tree = CART(training_set, n)
        
        bin_error = regression_error(f_tree, tune_df)
        
        bins_dict[n] = bin_error
        
    return bins_dict


# In[25]:


def cross_validation_unprune(data_subset_dict, n):
    """
    do five fold cross validatiaon with the training and test set for unpruning tree
    """
    
    test_result_unpruned = {}
    
    for num in range(1, 6):
        
        training_set, test_set = train_test_r(data_subset_dict, num)
        
        print("Training Set's average target value", np.mean(training_set['target']))
        print("Test Set's average target value", np.mean(test_set['target']))
        
        f_tree = CART(training_set, n)
        
        unpruned_error = regression_error(f_tree, test_set)
        
        test_result_unpruned[num] = unpruned_error
        
        test_average = sum(test_result_unpruned.values()) / len(test_result_unpruned)
        
    return test_result_unpruned, test_average


# In[26]:


def threshold_tuning(training_df, tune_df, n):
    """
    tune the threshold for early stopping
    """
    
    mse_dict = {}
    
    pre_mse = mse(training_df['target'])
    
    for t in [t * pre_mse for t in [0.3, 0.5, 0.7]]:
        f_tree = early_pruning(training_df, n, t)
        mse_dict[t] = regression_error(f_tree, tune_df)
        
    return mse_dict


# In[27]:


def early_pruning(data_df, n, threshold):
    
    if len(data_df['target'].unique()) == 1:
        target_val = data_df['target'].unique()[0]
        return TreeNode(target_val)
    
    
    else:
        target_val = np.mean(data_df['target'])
        f_tree = TreeNode(target_val)
        attribute_mse, attribute_selected, threshold_selected = tree_build(data_df, n)
        
        if attribute_mse < threshold:
            return f_tree
        
        else:
            #print('attribute', attribute_selected)
            f_tree.attribute = attribute_selected
            data_df_left, data_df_right = split_df(data_df, attribute_selected, threshold_selected)
            for val in ['left_' + str(threshold_selected), 'right_' + str(threshold_selected)]:
                if val == 'left_' + str(threshold_selected):
                    subset = data_df_left.loc[:, data_df_left.columns != attribute_selected]
                else:
                    subset = data_df_right.loc[:, data_df_right.columns != attribute_selected]
                
                subtree = CART(subset, n)
                subtree.parent = attribute_selected
                subtree.data_instances = subset
                f_tree.child[val] = subtree
        
    return f_tree
    


# In[28]:


def cross_validation_prune(data_subset_dict, tune_df, n):
    """
    do five fold cross validatiaon with the training and test set for unpruning tree
    """
    
    test_result_pruned = {}
    
    for num in range(1, 6):
        
        training_set, test_set = train_test_r(data_subset_dict, num)
        
        mse_dict = threshold_tuning(training_set, tune_df, n)
        
        
        val_list = list(mse_dict.values())
        
        key_list = list(mse_dict.keys())
        
        #print(mse_dict)
        
        threshold = key_list[val_list.index(min(mse_dict.values()))]
        
        
        print("Training Set's average target value", np.mean(training_set['target']))
        print("Test Set's average target value", np.mean(test_set['target']))
        
        f_tree = early_pruning(training_set, n, threshold)
        
        pruned_error = regression_error(f_tree, test_set)
        
        test_result_pruned[num] = pruned_error
        
        test_average = sum(test_result_pruned.values()) / len(test_result_pruned)
        
    return test_result_pruned, test_average


# In[29]:


def n_unprune(data_df):
    """
    Return the MSE with optimal n selected from tuning
    """
    n_dict = tune_bins(data_df)
    print(n_dict)
    val_list = list(n_dict.values())
    key_list = list(n_dict.keys())
    n = key_list[val_list.index(min(n_dict.values()))]
    print('Select n = ', n)
    
    return r_prune_unprune(data_df, n)
    


# In[30]:


def r_prune_unprune(data_df, n):
    tune_df, remaining_df = train_tune_test(data_df)
    data_subset_dict = create_fold_regression(data_df)
    test_result_unpruned, test_average_unpruned = cross_validation_unprune(data_subset_dict, n)
    test_result_pruned, test_average_pruned = cross_validation_prune(data_subset_dict, tune_df, n)
    return test_result_pruned, test_average_pruned, test_result_unpruned, test_average_unpruned,


# In[31]:


# – Provide sample outputs from one test set on one fold for a regression tree.
data_df = abalone
tune_df, remaining_df = train_tune_test(data_df)
data_subset_dict = create_fold_regression(data_df)
training_set, test_set = train_test_r(data_subset_dict, num = 1)
f_tree_unpruned = CART(training_set, n = 4)


# In[32]:


test_prediction_list = []
for i in range(len(test_set)):
    test_set_row = test_set.iloc[i, :]
    test_predictions = predict_numeric(test_set, f_tree_unpruned, test_set_row)
    test_prediction_list.append(test_predictions)


# In[33]:


test_prediction_list


# In[34]:


test_set['target'].to_list()


# In[35]:


# Show a sample regression tree without early stopping and with early stopping.

# without early stopping
print_branches(f_tree_unpruned)


# In[36]:


n = 4

mse_dict = threshold_tuning(training_set, tune_df, n)
        
val_list = list(mse_dict.values())

key_list = list(mse_dict.keys())

threshold = key_list[val_list.index(min(mse_dict.values()))]

print("Training Set's average target value", np.mean(training_set['target']))
print("Test Set's average target value", np.mean(test_set['target']))

f_tree_early = early_pruning(training_set, n, threshold)


# In[37]:


# with early stopping
print_branches(f_tree_early)


# In[38]:


# – Demonstrate the calculation of mean square error


# In[39]:


def mse(subset):
    """
    calculate the mean square error 
    """
    return np.mean((subset - np.mean(subset))**2)


# In[40]:


mse(training_set['target'])


# In[41]:


# – Demonstrate a decision being made to stop growing a subtree (early stopping).

def early_pruning(data_df, n, threshold):
    
    if len(data_df['target'].unique()) == 1:
        target_val = data_df['target'].unique()[0]
        return TreeNode(target_val)
    
    
    else:
        target_val = np.mean(data_df['target'])
        f_tree = TreeNode(target_val)
        attribute_mse, attribute_selected, threshold_selected = tree_build(data_df, n)
        
        print('MSE from the attribute', attribute_mse)
        print('threshold', threshold)
        if attribute_mse < threshold:
            return f_tree
        
        else:
            #print('attribute', attribute_selected)
            f_tree.attribute = attribute_selected
            data_df_left, data_df_right = split_df(data_df, attribute_selected, threshold_selected)
            for val in ['left_' + str(threshold_selected), 'right_' + str(threshold_selected)]:
                if val == 'left_' + str(threshold_selected):
                    subset = data_df_left.loc[:, data_df_left.columns != attribute_selected]
                else:
                    subset = data_df_right.loc[:, data_df_right.columns != attribute_selected]
                
                subtree = CART(subset, n)
                subtree.parent = attribute_selected
                subtree.data_instances = subset
                f_tree.child[val] = subtree
        
    return f_tree


# In[42]:


early_pruning(training_set, n, threshold)


# In[43]:


# – Demonstrate an example traversing a regression tree and a prediction being made at the leaf.

def predict_numeric(data_df, f_tree, val_set_row):
    """
    algorithm for making prediction
    """
    # if it is a leaf node, return the class label
    if len(f_tree.child) == 0:
        return f_tree.target
    
    else:
        # select the column values from the test set with the attribute in the tree
        a_val = val_set_row[f_tree.attribute]
        
        if data_df[f_tree.attribute].dtypes == 'object':
        
            if a_val == list(f_tree.child.keys())[0].split('_')[1]:
                return predict_numeric(data_df, f_tree.child[list(f_tree.child.keys())[0]], val_set_row)
            elif a_val != list(f_tree.child.keys())[0].split('_')[1]:
                return predict_numeric(data_df, f_tree.child[list(f_tree.child.keys())[1]], val_set_row)
                
        else:
            if a_val <= float(list(f_tree.child.keys())[0].split('_')[1]):
                return predict_numeric(data_df, f_tree.child[list(f_tree.child.keys())[0]], val_set_row)
            elif a_val > float(list(f_tree.child.keys())[0].split('_')[1]):
                return predict_numeric(data_df, f_tree.child[list(f_tree.child.keys())[1]], val_set_row)
        


# In[44]:


i = 0
val_set_row = test_set.iloc[i, :]
predict_numeric(test_set, f_tree_unpruned, val_set_row)


# In[45]:


test_set.iloc[0, :]


# In[46]:


# – Show the average performance over the five folds on a classification data set (with and without
# pruning) 
n_unprune(abalone)


# In[47]:


n_unprune(machine)


# In[48]:


n_unprune(forestfires)

