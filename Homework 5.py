# Classification

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import random
import math
from random import seed
seed(3)


# #### Classification:
# Breast Cancer
# 
# Glass
# 
# Soybean
# 
# #### Regression:
# Abalone
# 
# Computer Hardware
# 
# Forest Fires

# – Provide sample outputs from one test set showing performance on your feedforward networks.
# Show results for each of the cases where you have no hidden layers, one hidden layer, and two
# hidden layers.
# 
# – Show a sample model for the smallest of each of your three neural network types (i.e., zero hidden layer, one hidden layer, two hidden layers). This will consist of showing the weight matrices with the inputs/outputs of the layer labeled in some way.
# 
# – Demonstrate and explain how an example is propagated through a two hidden layer network. Be
# sure to show the activations at each layer being calculated correctly.
# 
# – Demonstrate the weight updates occurring on a two-layer network for each of the layers.
# 
# – Demonstrate the gradient calculation at the output for any one of your networks.
# 
# – Show the average performance over the five folds for one of the data sets for each of the three types of networks (i.e., zero hidden layer, one hidden layer, two hidden layers).

# In[2]:


def fillnan(df):
    """
    this function fill in the nan values with the average values for the attribute in the class
    """
    df = df.replace('?', np.nan)
    for c in df['class'].unique():
        df[df['class'] == c] = df[df['class'] == c].apply(lambda x: x.fillna(x.dropna().astype(int).mean()), axis = 0)
    return df


# In[3]:


def normalize_attr(df, col):
    """
    this function normalizes the numerical attributes so that it ranges from -1 to +1
    """
    scaler = MinMaxScaler(feature_range = (-1, 1))
    attr_cols = df.loc[:, df.columns != col].columns
    attr_df = pd.DataFrame(scaler.fit_transform(df.loc[:, df.columns != col]), columns = attr_cols)
    final_df = pd.concat([attr_df, df[col]], axis = 1)
    return final_df


# In[4]:


breast = pd.read_csv('breast-cancer-wisconsin.data', sep = ',', header = None)


# In[5]:


breast.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
             'Mitoses', 'class']


# In[6]:


breast.replace('?', np.nan).isna().sum()


# In[7]:


breast['class'] = breast['class'].map({2:0, 4:1})


# In[8]:


breast_df = fillnan(breast)


# In[9]:


breast_df = breast_df.drop(['Sample code number'], axis = 1)


# In[10]:


breast_df = normalize_attr(breast_df, 'class')


# In[11]:


breast_df.head()


# Glass Dataset:
# 
# • No missing values

# In[12]:


glass = pd.read_csv('glass.data', sep = ',', header = None)


# In[13]:


glass.columns = ['Id number', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']


# In[14]:


glass = glass.drop(['Id number'], axis = 1)


# In[15]:


glass_df = normalize_attr(glass, 'class')


# In[16]:


glass_df.head()


# In[17]:


glass_df['class'] = glass_df['class'].astype('category').cat.codes


# In[18]:


glass_df['class'].unique()


# Soybean Dataset:
# 
# •	No missing values

# In[19]:


soybean = pd.read_csv('soybean-small.data', sep = ',', header = None)


# In[20]:


soybean_df = soybean.rename(columns = {35 : 'class'}) 


# In[21]:


soybean.columns = ["date", "plant-stand", "precip", "temp", "hail", "crop-hist", "area-damaged", "severity","seed-tmt", "germination", "plant-growth", "leaves", "leafspots-halo", "leafspots-marg", "leafspot-size","leaf-shread", "leaf-malf", "leaf-mild", "stem", "lodging", "stem-cankers", "canker-lesio", "fruiting-bodie",
"external-decay", "mycelium", "int-discolor", "sclerotia", "fruit-pods", "fruit-spots", "seed", "mold-growth",
"seed-discolor", "seed-size", "shriveling", "roots", "class"]

selected_features = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity',
            'seed-tmt', 'germination', 'leaves', 'lodging', 'stem-cankers', 'canker-lesio',
            'fruiting-bodie', 'external-decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'roots','class']

soybean_df = soybean[selected_features]


# In[22]:


soybean_df['class'] = soybean_df['class'].astype('category').cat.codes


# In[23]:


soybean_df = normalize_attr(soybean_df, 'class')


# In[24]:


soybean_df.head()


# In[25]:


def tune_remaining(data_df, col):
    """
    take out 10% of the dataset for tuning and 90% for training and testing
    """
    class_df = pd.DataFrame(data_df.groupby([col])[col].count()).rename(columns = {col: 'class_proportion'})
    tune_df = pd.DataFrame()
    index_list = []
    
    for c in class_df.index.unique():
        c_df = data_df[data_df[col] == c].sample(n = int(np.ceil(len(data_df[data_df[col] == c]) * 0.1)))
        index_list.append(c_df.index.tolist())
        c_df = c_df.reset_index(drop = True) 
        tune_df = pd.concat([tune_df, c_df], axis = 0)
        
    tune_df = tune_df.reset_index(drop = True)
    
    flatten_list = [y for x in index_list for y in x]
    
    remaining_df = data_df[~data_df.index.isin(flatten_list)].reset_index(drop = True)
    
    return tune_df, remaining_df


# In[26]:


def create_fold_classification(data_df, col):
    """
    create 5 subsets for the classification dataset
    """
    # calculate the class proportion
    class_df = pd.DataFrame(data_df.groupby([col])[col].count()).rename(columns = {col: 'class_proportion'})
    
    # calculate the fold size
    class_df['fold_size'] = np.ceil(class_df['class_proportion']/5)
    class_df['fold_size'] = class_df['fold_size'].astype(int)
    
    # create a dictionary to store the data in each fold
    keys = [i for i in range(1, 6)]
    data_subset_dict = {k: [] for k in keys}
    
    for c in class_df.index.unique():
        class_subset = shuffle(data_df[data_df[col] == c]).reset_index(drop = True)

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


# In[27]:


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


# In[28]:


def neural_net_weights(num_inputs, num_hidden_layers, num_nodes, num_outputs):
    """
    initialize the weights for the neural network
    """
    
    network = []
    output_layer = []
    
    # no hidden layer
    if num_hidden_layers == 0: 
        for n in range(num_outputs):
            weight_dict = {}
            weight_dict['weights'] = [random.random() for i in range(num_inputs + 1)]
            output_layer.append(weight_dict)
        network.append(output_layer)
    else:
        # neural network with hidden layers
        
        counter = 0
        
        for l in range(num_hidden_layers):
            if counter == 0:
            # for the first hidden layer
                hidden_layer = []
                for n in range(num_nodes):
                    weight_dict = {}
                    weight_dict['weights'] = [random.random() for i in range(num_inputs + 1)]
                    hidden_layer.append(weight_dict)
                network.append(hidden_layer)
                counter += 1
            else:
                hidden_layer = []
                for n in range(num_nodes):
                    weight_dict = {}
                    weight_dict['weights'] = [random.random() for i in range(num_nodes + 1)]
                    hidden_layer.append(weight_dict)
                network.append(hidden_layer)
    
        for n in range(num_outputs):
            weight_dict = {}
            weight_dict['weights'] = [random.random() for i in range(num_nodes + 1)]
            output_layer.append(weight_dict)
        network.append(output_layer)
    
    return network


# In[29]:


def feedforward(inputs, network):
    """
    feedforward propagation
    """
    for layer in network:
        updated_inputs = []
        for node in layer:
            #print('node', node)
            #print(np.dot(inputs, node['weights']))
            weighted_sum = float(np.dot(inputs, node['weights']))
            #print('node weights', node['weights'])
            #print(weighted_sum)
            sigmoid_output = 1 / (1 + math.exp(-weighted_sum))
            node['output'] = sigmoid_output
            #print(node['output'])
            updated_inputs.append(node['output'])
        inputs = np.array(updated_inputs)
        inputs = np.append(inputs, 1)
    return inputs[:-1]
    


# In[30]:


def backpropagate(network, actual_class):
    
    """
    backpropagation process
    """
    
    for i in range(len(network)-1, -1, -1):
        error_list = []
        
        # for the output layer, calculate the error
        if i == len(network) - 1:
            for n, node in enumerate(network[i]):
                error_list.append(actual_class[n] - node['output'])
        else:
            # backpropogation in the hidden layer
            for n, node in enumerate(network[i]):
                error = 0
                for node in network[i + 1]:
                    error += (node['weights'][n] * node['delta'])
                error_list.append(error)
        
        # backpropagation in the output layer
        for n, node in enumerate(network[i]):
            node['delta'] = error_list[n] * (node['output'] * (1-node['output']))


# In[31]:


def weights_update(network, inputs, lr):
    """
    update the weights in the network
    """
    for i in range(len(network)):
    
        # if we are not on the first layer
        if i != 0:
            inputs = [node['output'] for node in network[i - 1]]

        for n, node in enumerate(network[i]):

            for j in range(len(inputs)):

                node['weights'][j] += lr * node['delta'] * inputs[j]
        


# In[32]:


def neural_network_training(network, training_set, lr, num_epoch):
    """
    neural network training 
    """
    for epoch in range(num_epoch):
        for i in range(len(training_set)):
            X = np.array(training_set.loc[i, training_set.columns != 'class'])
            X_input = np.append(X, 1)
            y = training_set.loc[i, 'class']
            num_outputs = len(training_set['class'].unique())
            actual_class = [0 for i in range(num_outputs)]
            actual_class[int(y)] = 1
            feedforward(X_input, network)
            backpropagate(network, actual_class)
            weights_update(network, X_input, lr)


# In[33]:


def neural_network_test(test_set, network):
    """
    get test accuracy for neural network
    """
    predictions_list = []
    for i in range(len(test_set)):
        X = np.array(test_set.loc[i, test_set.columns != 'class'])
        X_input = np.append(X, 1)
        y = test_set.loc[i, 'class']
        outputs = feedforward(X_input, network)
        predictions_list.append(np.argmax(outputs))
    counter = 0
    for i in range(len(predictions_list)):
        if predictions_list[i] == test_set['class'].to_list()[i]:
            counter += 1

    return counter / len(predictions_list)
    


# In[34]:


def nn_training_parameter(training_set, tuning_set, num_hidden_layers):
    """
    Hyperparameter tuning process
    """
    
    if num_hidden_layers == 0:
        num_nodes_list = [0]
    else:
        num_nodes_list = [5, 8, 10]
        
    lr_list = [0.1, 0.01, 0.001]
    num_epochs_list = [10, 30, 50]
    
    output_list = []
    
    for lr in lr_list:
        for num_epoch in num_epochs_list:
            for num_nodes in num_nodes_list:
                num_inputs = len(training_set.loc[0, training_set.columns != 'class'])
                num_outputs = len(training_set['class'].unique())
                network = neural_net_weights(num_inputs, num_hidden_layers, num_nodes, num_outputs)
                neural_network_training(network, training_set, lr, num_epoch)
                accuracy = neural_network_test(tuning_set, network)
                output_list.append([lr, num_epoch, num_nodes, accuracy])
    output_df = pd.DataFrame(output_list, columns = ['Learning Rate', '# Epochs', '# Nodes', 'Accuracy'])
    
    return output_df
    


# In[35]:


def optimal_parameter(output_df):
    """
    find the best hyperparameters after tuning and return its accuracy on the test set
    """
    
    optimal_lr = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'Learning Rate']
    
    optimal_epoch = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, '# Epochs']
    
    optimal_nodes = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, '# Nodes']

    return optimal_lr, optimal_epoch, optimal_nodes


# In[36]:


def cross_validation(data_df, num_hidden_layers):
    
    """
    cross validation process
    """
    
    tune_df, remaining_df = tune_remaining(data_df, 'class')
    
    data_subset_dict = create_fold_classification(remaining_df, 'class')

    test_result = []
    
    cv_accuracy = {}
    
    for num in range(1,6):
        
        training_set, test_set = train_test_c(data_subset_dict, num)
        
        output_df = nn_training_parameter(training_set, tune_df, num_hidden_layers)
        
        optimal_lr, optimal_epoch, optimal_nodes = optimal_parameter(output_df)
        
        print('Fold Number: ', num)
        
        print('Optimal learning rate, epoches, nodes: ', optimal_lr, optimal_epoch, optimal_nodes)
        
        num_inputs = len(training_set.loc[0, training_set.columns != 'class'])
        
        num_outputs = len(training_set['class'].unique())
        
        network = neural_net_weights(num_inputs, num_hidden_layers, optimal_nodes, num_outputs)
        
        neural_network_training(network, training_set, optimal_lr, optimal_epoch)
        
        accuracy = neural_network_test(test_set, network)
        
        test_result.append(accuracy)
        
        cv_accuracy[num] = accuracy
        
    test_average = sum(test_result) / len(test_result)
        
    return cv_accuracy, test_average
        


# In[37]:


for num_hidden_layers in range(3):
    print('Number of Hidden Layers: ', num_hidden_layers)
    print(cross_validation(breast_df, num_hidden_layers))


# In[38]:


for num_hidden_layers in range(3):
    print('Number of Hidden Layers: ', num_hidden_layers)
    print(cross_validation(glass_df, num_hidden_layers))


# In[39]:


for num_hidden_layers in range(3):
    print('Number of Hidden Layers: ', num_hidden_layers)
    print(cross_validation(soybean_df, num_hidden_layers))



# Regression

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import random
import math
from random import seed
seed(3)


# #### Classification:
# Breast Cancer
# 
# Glass
# 
# Soybean
# 
# #### Regression:
# Abalone
# 
# Computer Hardware
# 
# Forest Fires

# – Provide sample outputs from one test set showing performance on your feedforward networks.
# Show results for each of the cases where you have no hidden layers, one hidden layer, and two
# hidden layers.
# 
# – Show a sample model for the smallest of each of your three neural network types (i.e., zero hidden layer, one hidden layer, two hidden layers). This will consist of showing the weight matrices with the inputs/outputs of the layer labeled in some way.
# 
# – Demonstrate and explain how an example is propagated through a two hidden layer network. Be
# sure to show the activations at each layer being calculated correctly.
# 
# – Demonstrate the weight updates occurring on a two-layer network for each of the layers.
# 
# – Demonstrate the gradient calculation at the output for any one of your networks.
# 
# – Show the average performance over the five folds for one of the data sets for each of the three types of networks (i.e., zero hidden layer, one hidden layer, two hidden layers).

# In[2]:


def normalize_attr(df, col):
    """
    this function normalizes the numerical attributes so that it ranges from -1 to +1
    """
    scaler = MinMaxScaler(feature_range = (-1, 1))
    attr_cols = df.loc[:, df.columns != col].columns
    attr_df = pd.DataFrame(scaler.fit_transform(df.loc[:, df.columns != col]), columns = attr_cols)
    final_df = pd.concat([attr_df, df[col]], axis = 1)
    return final_df


# Abalone Dataset

# In[3]:


abalone = pd.read_csv('abalone.data', sep = ',', header = None)


# In[4]:


abalone.head()


# In[5]:


abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'target']


# In[6]:


ohe = OneHotEncoder()
X = pd.DataFrame(abalone['Sex'])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]
sex_df = pd.DataFrame(X)


# In[7]:


attribute_df = abalone.loc[:, abalone.columns != 'Sex']


# In[8]:


abalone_attribute = normalize_attr(attribute_df, 'target')


# In[9]:


abalone_df = pd.concat([sex_df, abalone_attribute], axis = 1)


# In[10]:


abalone_df


# Computer Hardware Dataset

# In[11]:


machine = pd.read_csv('machine.data', header = None)


# In[12]:


machine.columns = ['Vendor name', 'Model Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'target', 'ERP']


# In[13]:


machine = machine.drop(['Model Name', 'ERP'], axis = 1)


# In[14]:


ohe = OneHotEncoder()
X = pd.DataFrame(machine['Vendor name'])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]
vendor_df = pd.DataFrame(X)


# In[15]:


attribute_df = machine.loc[:, machine.columns != 'Vendor name']


# In[16]:


machine_attribute = normalize_attr(attribute_df, 'target')


# In[17]:


machine_df = pd.concat([vendor_df, machine_attribute], axis = 1)


# In[18]:


machine_df


# Forestfire Dataset

# In[19]:


forestfires = pd.read_csv('forestfires.data')


# In[20]:


forestfires = forestfires.rename(columns = {'area': 'target'})


# In[21]:


ohe = OneHotEncoder()
X = pd.DataFrame(forestfires[['X', 'Y', 'month', 'day']])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]
forestfires_df = pd.DataFrame(X)


# In[22]:


attribute_df = forestfires[forestfires.columns.difference(['X', 'Y', 'month', 'day'])]


# In[23]:


forestfires_attribute = normalize_attr(attribute_df, 'target')


# In[24]:


forestfires_df = pd.concat([forestfires_df, forestfires_attribute], axis = 1)


# In[25]:


forestfires_df


# In[26]:


def train_tune_test(df):
    
    """
    use the original dataset to create training set, test set and tuning set
    """
    
    # get the tuning set
    X_tune = df.sample(frac = 0.1)
    
    # get the remaining set
    X_remaining = df[~df.index.isin(X_tune.index)]
    
    X_tune = X_tune.reset_index(drop = True)
    
    X_remaining = X_remaining.reset_index(drop = True)
    
    return X_tune, X_remaining


# In[27]:


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


# In[28]:


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


# In[29]:


def neural_net_weights(num_inputs, num_hidden_layers, num_nodes, num_outputs):
    """
    initialize the weights for the neural network
    """
    
    network = []
    output_layer = []
    
    # no hidden layer
    if num_hidden_layers == 0: 
        for n in range(num_outputs):
            weight_dict = {}
            weight_dict['weights'] = [random.uniform(-1, 1) for i in range(num_inputs + 1)]
            output_layer.append(weight_dict)
        network.append(output_layer)
    else:
        # neural network with hidden layers
        
        counter = 0
        
        for l in range(num_hidden_layers):
            if counter == 0:
            # for the first hidden layer
                hidden_layer = []
                for n in range(num_nodes):
                    weight_dict = {}
                    weight_dict['weights'] = [random.uniform(-1, 1) for i in range(num_inputs + 1)]
                    hidden_layer.append(weight_dict)
                network.append(hidden_layer)
                counter += 1
            else:
                hidden_layer = []
                for n in range(num_nodes):
                    weight_dict = {}
                    weight_dict['weights'] = [random.uniform(-1, 1) for i in range(num_nodes + 1)]
                    hidden_layer.append(weight_dict)
                network.append(hidden_layer)
    
        for n in range(num_outputs):
            weight_dict = {}
            weight_dict['weights'] = [random.uniform(-1, 1) for i in range(num_nodes + 1)]
            output_layer.append(weight_dict)
        network.append(output_layer)
    
    return network


# In[30]:


def feedforward_regression(inputs, network):
    """
    feedforward propagation
    """
    for i, layer in enumerate(network):
        if i != len(network) - 1:
            updated_inputs = []
            for node in layer:
                weighted_sum = float(np.dot(inputs, node['weights']))
                if weighted_sum <= -500:
                    node['output'] = 0
                else:
                    sigmoid_output = 1 / (1 + math.exp(-weighted_sum))
                    node['output'] = sigmoid_output
                updated_inputs.append(node['output'])
            inputs = np.array(updated_inputs)
            inputs = np.append(inputs, 1)
        else:
            for node in layer:
                weighted_sum = float(np.dot(inputs, node['weights']))
                node['output'] = weighted_sum
            return weighted_sum


# In[31]:


def backpropagate_regression(network, actual_target):
    """
    backpropagation process
    """    
    for i in range(len(network)-1, -1, -1):
        error_list = []

        # for the output layer, calculate the error
        if i == len(network) - 1:
            delta = (actual_target - network[len(network) - 1][0]['output'])
            error_list.append(delta)
            for n, node in enumerate(network[i]):
                node['delta'] = delta
        else:
            # backpropogation in the hidden layer
            for n, node in enumerate(network[i]):
                #print('current node', node)
                error = 0
                for node in network[i + 1]:
                    #print('previous node', node)
                    error += (node['weights'][n] * node['delta'])
                    #print('error', error)

                error_list.append(error)

            #print('error list', error_list)
            for n, node in enumerate(network[i]):
                #print('node output', error_list[n], (node['output'], (1-node['output'])))
                node['delta'] = error_list[n] * (node['output'] * (1-node['output']))
                


# In[32]:


def weights_update(network, inputs, lr):
    """
    update the weights in the network
    """
    for i in range(len(network)):
    
        # if we are not on the first layer
        if i != 0:
            inputs = [node['output'] for node in network[i - 1]]

        for n, node in enumerate(network[i]):

            for j in range(len(inputs)):

                node['weights'][j] += lr * node['delta'] * inputs[j]
        


# In[33]:


def neural_network_training(network, training_set, lr, num_epoch):    
    """
    neural network training 
    """
    for epoch in range(num_epoch):
        #print(epoch)
        for i in range(len(training_set)):
            X = np.array(training_set.loc[i, training_set.columns != 'target'])
            X_input = np.append(X, 1)
            #print(X_input)
            actual_target = training_set.loc[i, 'target']
            #print(actual_target)
            feedforward_regression(X_input, network)
            backpropagate_regression(network, actual_target)
            #print(network)
            weights_update(network, X_input, lr)


# In[34]:


def neural_network_test(test_set, network):    
    """
    get test accuracy for neural network
    """
    predictions_list = []
    for i in range(len(test_set)):
        X = np.array(test_set.loc[i, test_set.columns != 'target'])
        X_input = np.append(X, 1)
        actual_target = test_set.loc[i, 'target']
        outputs = feedforward_regression(X_input, network)
        predictions_list.append(outputs)
    
    mse = 0
    for i in range(len(predictions_list)):
        mse += (predictions_list[i] - test_set['target'].to_list()[i])**2
    
    return mse/len(predictions_list)


# In[35]:


def nn_training_parameter(training_set, tuning_set, num_hidden_layers):
    """
    Hyperparameter tuning process
    """
    
    if num_hidden_layers == 0:
        num_nodes_list = [0]
    else:
        num_nodes_list = [5, 8, 10]
        
    lr_list = [0.1, 0.01, 0.001]
    num_epochs_list = [10, 30, 50]
    
    output_list = []
    
    for lr in lr_list:
        for num_epoch in num_epochs_list:
            for num_nodes in num_nodes_list:
                num_inputs = len(training_set.loc[0, training_set.columns != 'target'])
                num_outputs = len(training_set['target'].unique())
                network = neural_net_weights(num_inputs, num_hidden_layers, num_nodes, num_outputs)
                neural_network_training(network, training_set, lr, num_epoch)
                mse = neural_network_test(tuning_set, network)
                output_list.append([lr, num_epoch, num_nodes, mse])
    output_df = pd.DataFrame(output_list, columns = ['Learning Rate', '# Epochs', '# Nodes', 'MSE'])
    
    return output_df
    


# In[36]:


def optimal_parameter(output_df):
    """
    find the best hyperparameters after tuning and return its MSE on the test set
    """
    
    optimal_lr = output_df[output_df['MSE'] == output_df['MSE'].min()].reset_index().loc[0, 'Learning Rate']
    
    optimal_epoch = output_df[output_df['MSE'] == output_df['MSE'].min()].reset_index().loc[0, '# Epochs']
    
    optimal_nodes = output_df[output_df['MSE'] == output_df['MSE'].min()].reset_index().loc[0, '# Nodes']

    return optimal_lr, optimal_epoch, optimal_nodes


# In[37]:


def cross_validation(data_df, num_hidden_layers):
    
    """
    cross validation process
    """
    
    tune_df, remaining_set = train_tune_test(data_df)
    
    data_subset_dict = create_fold_regression(remaining_set)

    test_result = []
    
    cv_mse = {}
    
    for num in range(1,6):
        
        training_set, test_set = train_test_r(data_subset_dict, num)
        
        output_df = nn_training_parameter(training_set, tune_df, num_hidden_layers)
        
        optimal_lr, optimal_epoch, optimal_nodes = optimal_parameter(output_df)
        
        print('Fold Number: ', num)
        
        print('Optimal learning rate, epoches, nodes: ', optimal_lr, optimal_epoch, optimal_nodes)
        
        num_inputs = len(training_set.loc[0, training_set.columns != 'target'])
        
        num_outputs = len(training_set['target'].unique())
        
        network = neural_net_weights(num_inputs, num_hidden_layers, optimal_nodes, num_outputs)
        
        neural_network_training(network, training_set, optimal_lr, optimal_epoch)
        
        mse = neural_network_test(test_set, network)
        
        test_result.append(mse)
        
        cv_mse[num] = mse
        
    test_average = sum(test_result) / len(test_result)
        
    return cv_mse, test_average
        


# In[ ]:


for num_hidden_layers in range(3):
    print('Number of Hidden Layers: ', num_hidden_layers)
    print(cross_validation(abalone_df, num_hidden_layers))


# In[ ]:


for num_hidden_layers in range(3):
    print('Number of Hidden Layers: ', num_hidden_layers)
    print(cross_validation(machine_df, num_hidden_layers))


# In[ ]:


for num_hidden_layers in range(3):
    print('Number of Hidden Layers: ', num_hidden_layers)
    print(cross_validation(forestfires_df, num_hidden_layers))

