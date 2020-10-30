#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math


# Preprocess the data
# * Load the datasets
# * Assigns columns
# * Data Imputation: Mean Imputation
# * Binning/Discretization
# * One-Hot Encoding

# Breast Cancer Dataset:
# 
# •	Missing values handling: separate the observations with missing values based on their classes and replace the missing values with the average of the attribute in the class
# 

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


breast = pd.read_csv('breast-cancer-wisconsin.data', sep = ',', header = None)


# In[4]:


breast.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
             'Mitoses', 'class']


# In[5]:


breast


# In[6]:


def binning(df, col, bins):
    
    """
    create bins for the mult-value discrete variables
    """
    if len(df[col].unique()) < bins:
        return df[col]
    
    else:
    
        df[col] = df[col].astype(float)

        # find the minimum value
        min_val = df[col].min()

        # find the maximum value
        max_val = df[col].max()


        # create a dictionary to store the bin numbers and its corresponding intervals
        i = min_val
        k = 0
        bin_list = [i for i in range(bins)]
        bin_dict = {}

        while k < len(bin_list):
            bin_dict[k] = [i]
            width_size = (max_val - min_val) / bins

            buffer = width_size + 0.1 * width_size
            i += buffer

            bin_dict[k].append(i)
            k+=1

        bin_list = []
        for val in df[col]:
            for i in bin_dict:
                if (bin_dict[i][0] <= val < bin_dict[i][1]) == True:
                    bin_list.append(i)
        df[col] = bin_list

        return df[col]


# In[7]:


def discritization(df):
    """
    discritization for multi-value discrete variables
    """
    for col in df.columns:
        df[col] = binning(df, col, bins)
    
    return df


# In[8]:


def one_hot_encoding(df, col):
    """
    map the non-binary categorical variable to one-hot coding
    
    """
    one_hot_encoded = list()
    
    for val in df[col]:

        if len(df[col].unique()) == 1:
            z_list = 1
        else:
            z_list = [0 for _ in range(len(df[col].unique()))] 
            z_list[val] = 1
        one_hot_encoded.append(z_list)
    
    one_hot_df = pd.DataFrame(one_hot_encoded)
    one_hot_df.columns = [str(col) + '_' + str(i) for i in range(len(one_hot_df.columns))]
    return one_hot_df


# In[9]:


def one_hot_encoding_df(df, o_df):
    """
    map the non-binary categorical variable to one-hot coding for all the real-valued features
    
    """
    final_df = pd.DataFrame()
    
    for col in df.columns:
        one_hot_df = one_hot_encoding(df, col)
        final_df = pd.concat([final_df, one_hot_df], axis = 1)
    
    final_df['class'] = o_df['class'].astype('category').cat.codes
    return final_df


# In[10]:


def attribute_mod(attribute_df, df):
    """
    this function modifies the attribute values, if the attribute values > the mean, assigns 1, otherwise, assigns 0
    """
    for col in attribute_df.columns:
        attribute_df[col] = np.where(attribute_df[col].astype(float) > np.mean(attribute_df[col].astype(float)), 1, 0)
    df['class'] = df['class'].astype('category').cat.codes
    final_df = pd.concat([attribute_df, df['class']], axis = 1)
    
    return final_df


# In[11]:


breast.shape


# In[12]:


breast.groupby(['class'])['class'].count()


# In[13]:


breast.replace('?', np.nan).isna().sum()


# In[14]:


# define the number of bins
bins = 3


# In[15]:


breast_df = fillnan(breast)
bin_breast_df = breast_df.iloc[:, 1: -1]
final_breast = discritization(bin_breast_df)
final_breast.head()


# In[16]:


one_hot_breast = one_hot_encoding_df(final_breast, breast_df)


# In[17]:


one_hot_breast


# Glass Dataset:
# 
# •	No missing values

# In[18]:


glass = pd.read_csv('glass.data', sep = ',', header = None)


# In[19]:


glass.columns = ['Id number', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']


# In[20]:


glass.head()


# In[21]:


glass.shape


# In[22]:


glass.groupby(['class'])['class'].count()


# In[23]:


glass.replace('?', np.nan).isnull().values.any()


# In[24]:


attribute_glass = glass.iloc[:, 1: -1]
final_glass = attribute_mod(attribute_glass, glass)


# Iris Dataset:
# 
# •	No missing values

# In[25]:


iris = pd.read_csv('iris.data', sep = ',', header = None)


# In[26]:


iris.head()


# In[27]:


iris.replace('?', np.nan).isnull().values.any()


# In[28]:


iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']


# In[29]:


iris.groupby(['class'])['class'].count()


# In[30]:


iris.shape


# In[31]:


bin_iris_df = iris.iloc[:, 1: -1]

final_iris = discritization(bin_iris_df)
final_iris.head()

one_hot_iris = one_hot_encoding_df(final_iris, iris)


# In[32]:


one_hot_iris


# Soybean Dataset:
# 
# •	No missing values

# In[33]:


soybean = pd.read_csv('soybean-small.data', sep = ',', header = None)


# In[34]:


soybean.head()


# In[35]:


soybean.replace('?', np.nan).isnull().values.any()


# In[36]:


soybean = soybean.rename(columns = {35 : 'class'}) 


# In[37]:


soybean.shape


# In[38]:


soybean.groupby(['class'])['class'].count()


# In[39]:


attribute_soybean = soybean.iloc[:, 0: -1]
final_soybean = attribute_mod(attribute_soybean, soybean)


# Vote Dataset:
# 
# •	Replace all ‘y’ with 1 and ‘n’ with 0 
# 
# •	Missing values handling: separate the observations with missing values based on their classes and replace the missing values with the average of the attribute in the class

# In[40]:


vote = pd.read_csv('house-votes-84.data', sep = ',', header = None)


# In[41]:


vote.columns = ['class', 'handicapped-infants', 'water-project-cost-sharing',
               'adoption-of-the-budget-resolution', 'physician-fee-freeze', 
               'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
               'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
               'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
               'export-administration-act-south-africa']


# In[42]:


vote.shape


# In[43]:


vote = vote.replace('?', np.nan)


# In[44]:


vote = vote.replace('n', 0)


# In[45]:


vote = vote.replace('y', 1)


# In[46]:


vote.head()


# In[47]:


vote.groupby(['class'])['class'].count()


# In[48]:


vote['class'] = vote['class'].astype('category').cat.codes


# In[49]:


vote_df = fillnan(vote)
attribute_vote = vote_df.iloc[:, 1:]
final_vote = attribute_mod(attribute_vote, vote_df)


# ### Winnow-2 Algorithm

# In[50]:


def winnow_2(df, theta, alpha, class_type):
    """
    Winnow-2 Training Algorithm
    """
    if class_type == 'binary':
        attributes = df.iloc[:, :-1]
    elif class_type == 'multiclass':
        attributes = df.iloc[:, :-2]
    
    num_attributes = attributes.shape[-1]
    # initialize all the weights for the attributes to 1
    weights = np.ones(num_attributes)
    num_attributes = attributes.shape[-1]
    
    # calculate the weighted sum using the weights and attributes
    for row in range(0, attributes.shape[0]):
        weighted_sum = np.dot(np.array(attributes.iloc[row, :]), weights)
        # if the weighted sum is greater than the threshold, the learner assigns 1 to the data instance. Otherwise, it assigns 0. 
        if weighted_sum > theta:
            prediction = 1
        else:
            prediction = 0
        
        # do nothing
        if prediction == df.loc[row, 'class']:
            weights = weights
        
    
        #promotion: if the learner's prediction = 0 and the true label = 1, 
        
        #if the true label = 1, weight for the ith attribute will be updated with alpha * weight 
        
        #otherwise, weight remains the same
        
        elif (prediction == 0 and df.loc[row, 'class'] == 1):
            weights[np.where(np.array(attributes.iloc[row, :]) == 1)] = weights[np.where(np.array(attributes.iloc[row, :]) == 1)] * alpha
        

        #demotion: if the learner's prediction = 1 and the true label = 0,
        
        #if the true label = 1, weight for the ith attribute will be updated with weight/alpha 
        
        #otherwise, weight remains the same
    
        elif (prediction == 1 and df.loc[row, 'class'] == 0):
            weights[np.where(np.array(attributes.iloc[row, :]) == 1)] = weights[np.where(np.array(attributes.iloc[row, :]) == 1)]/alpha
    return weights


# In[51]:


def winnow_test(df, weights_train, theta, class_type):
    """
    Winnow-2 Testing Algorithm
    """
    if class_type == 'binary':
        attributes = df.iloc[:, :-1]
    elif class_type == 'multiclass':
        attributes = df.iloc[:, :-2]
    
    num_attributes = attributes.shape[-1]
    weights = np.ones(num_attributes)
    num_attributes = attributes.shape[-1]

    df['Prediction'] = np.nan
    
    # apply the weighted sum trained from the model to the test set
    for row in range(0, attributes.shape[0]):
        weighted_sum = np.dot(np.array(attributes.iloc[row, :]), weights_train)
        
        # if the weighted sum is greater than the threshold, the learner’s prediction is 1
        if weighted_sum > theta:
            prediction = 1
            df.loc[row, 'Prediction'] = prediction
        # otherwise, it is 0. 
        else:
            prediction = 0
            df.loc[row, 'Prediction'] = prediction
    accuracy = df[df['class'] == df['Prediction']].shape[0] / df.shape[0]
    
    return accuracy


# In[52]:


def train_tune_test(df):
    
    """
    use the original dataset to create training set, test set and tuning set
    """
    
    # get the tuning set
    X_tune = df.sample(frac = 0.1)
    
    # get the training set
    X_train = df[~df.index.isin(X_tune.index)].sample(frac = (2/3))

    # get the test set
    X_test = df[~(df.index.isin(X_train.index)) & ~(df.index.isin(X_tune.index))]
    
    X_tune = X_tune.reset_index(drop = True)
    
    X_train = X_train.reset_index(drop = True)
    
    X_test = X_test.reset_index(drop = True)
    
    return X_tune, X_train, X_test


# In[53]:


def algorithm_training(X_tune, X_train, class_type):
    """
    Winnow-2 Algorithm that tests different theta and alpha value
    """
    
    theta_list = [0.1, 0.25, 0.5, 0.75]

    alpha_list = [2, 3, 4, 5]

    output_list = []
    for theta in theta_list:
        for alpha in alpha_list: 
            X_train = X_train.reset_index(drop = True)
            weights_train = winnow_2(X_train, theta, alpha, class_type)
            output_list.append([theta, alpha, weights_train, winnow_test(X_tune, weights_train, theta, class_type)])
            X_tune = X_tune.drop('Prediction', axis = 1)
    outout_df = pd.DataFrame(output_list, columns = ['Theta', 'Alpha', 'Weight_Trained', 'Accuracy'])
            
    return outout_df


# In[54]:


def optimal_parameter(X_train, X_test, output_df, class_type):
    """
    find the best hyperparameters after tuning and return the accuracy on the test set
    """
    optimal_theta = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'Theta']

    optimal_alpha = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'Alpha']
    
    X_train = X_train.reset_index(drop = True)
    
    weights_train = winnow_2(X_train, optimal_theta, optimal_alpha, class_type)
    
    return optimal_theta, optimal_alpha, winnow_test(X_test, weights_train, optimal_theta, class_type)


# In[55]:


def train_tune_test_multi_class(df):
    
    """
    use the original dataset to create training set, test set and tuning set for multiclass
    """
    
    df = df.rename(columns = {'class': 'class_multiple'})
    X_tune, X_train, X_test = train_tune_test(df)
    return X_tune, X_train, X_test


# In[56]:


def winnow_multiclass_training(X_train, theta, alpha):
    """
    winnow-2 training algorithm for multiclass dataset
    """    
    weights_train_vector = {}
    
    # create one classifier for each class in the dataset
    for i in range(0, len(X_train['class_multiple'].unique())):
        X_train['class'] = np.where(X_train['class_multiple'] == i, 1, 0)
        weights_train = winnow_2(X_train, theta, alpha, 'multiclass')
        weights_train_vector[i] = weights_train
    # train the model for each class and get the weights vector, which results in K weights vectors assuming we have K classes
    return weights_train_vector


# In[57]:


def winnow_multiclass_test(X_test, weights_train_vector):
    """
    winnow-2 testing algorithm for multiclass dataset
    """
    
    attributes = X_test.iloc[:, :-1]
    num_attributes = attributes.shape[-1]
    weights = np.ones(num_attributes)
    num_attributes = attributes.shape[-1]

    weight_list_df = []

    # apply the K weights vectors to the test set, for each observation in the test set, we have K weighted sum.  
    for key, value in weights_train_vector.items():
        weight_list = []
        for row in range(0, attributes.shape[0]):
            weighted_sum = np.dot(np.array(attributes.iloc[row, :]), weights_train_vector[key])
            weight_list.append(weighted_sum)
        weight_list_df.append(weight_list)

    weight_df = pd.DataFrame(weight_list_df).T
    # the observation is classified as the class with the highest weighted sum
    X_test['Prediction']= pd.DataFrame(weight_df.eq(weight_df.max(1), axis = 0).dot(weight_df.columns), columns = ['Prediction'])
    accuracy = X_test[X_test['class_multiple'] == X_test['Prediction']].shape[0] / X_test.shape[0]

    return accuracy


# In[58]:


def algorithm_training_multi_class(X_tune, X_train):
    
    """
    winnow-2 algorithm for multiclass dataset that tests different theta and alpha values
    """
    
    theta_list = [0.1, 0.25, 0.5, 0.75]
    alpha_list = [2, 3, 4, 5]

    output_list = []
    for theta in theta_list:
        for alpha in alpha_list:
            X_train = X_train.reset_index(drop = True)
            weights_train_vector = winnow_multiclass_training(X_train, theta, alpha)
            output_list.append([theta, alpha, weights_train_vector, winnow_multiclass_test(X_tune, weights_train_vector)])
            X_tune = X_tune.drop('Prediction', axis = 1)

    output_df = pd.DataFrame(output_list, columns = ['Theta', 'Alpha', 'Weight_Trained', 'Accuracy'])
    
    return output_df


# In[59]:


def optimal_parameter_multi_class(X_train, X_test, output_df):
    
    """
    find the best hyperparameters after tuning and return the accuracy on the test set for multiclass
    """
    
    optimal_theta = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'Theta']

    optimal_alpha = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'Alpha']

    weights_train_vector = winnow_multiclass_training(X_train, optimal_theta, optimal_alpha)

    return optimal_theta, optimal_alpha, winnow_multiclass_test(X_test, weights_train_vector)


# In[60]:


# Breast Cancer Dataset:

X_tune, X_train, X_test = train_tune_test(one_hot_breast)
breast_accuracy_df = algorithm_training(X_tune, X_train, 'binary')


# In[61]:


X_tune


# In[62]:


X_train


# In[63]:


X_test


# In[64]:


breast_accuracy_df


# In[65]:


optimal_parameter(X_train, X_test, breast_accuracy_df, 'binary')


# In[66]:


# Glass Dataset:
X_tune, X_train, X_test = train_tune_test_multi_class(final_glass)
glass_accuracy = algorithm_training_multi_class(X_tune, X_train)


# In[67]:


glass_accuracy


# In[68]:


optimal_parameter_multi_class(X_train, X_test, glass_accuracy)


# In[69]:


# Iris Dataset:

X_tune, X_train, X_test = train_tune_test_multi_class(one_hot_iris)
iris_accuracy = algorithm_training_multi_class(X_tune, X_train)


# In[70]:


iris_accuracy


# In[71]:


optimal_parameter_multi_class(X_train, X_test, iris_accuracy)


# In[72]:


# Soybean Dataset:
X_tune, X_train, X_test = train_tune_test_multi_class(final_soybean)
soybean_accuracy = algorithm_training_multi_class(X_tune, X_train)


# In[73]:


soybean_accuracy


# In[74]:


optimal_parameter_multi_class(X_train, X_test, soybean_accuracy)


# In[75]:


# Vote Dataset:

X_tune, X_train, X_test = train_tune_test(final_vote)
final_vote_accuracy = algorithm_training(X_tune, X_train, 'binary')


# In[76]:


final_vote_accuracy


# In[77]:


optimal_parameter(X_train, X_test, final_vote_accuracy, 'binary')


# ### Naive Bayes Algorithm

# In[78]:


def train_tune_test(df):
    
    """
    use the original dataset to create training set, test set and tuning set
    """
    
    # get the tuning set
    X_tune = df.sample(frac = 0.1)
    
    # get the training set
    X_train = df[~df.index.isin(X_tune.index)].sample(frac = (2/3))

    # get the test set
    X_test = df[~(df.index.isin(X_train.index)) & ~(df.index.isin(X_tune.index))]
    
    X_tune = X_tune.reset_index(drop = True)
    
    X_train = X_train.reset_index(drop = True)
    
    X_test = X_test.reset_index(drop = True)
    
    return X_tune, X_train, X_test


# In[79]:


def naive_bayes(m, p, X_train, X_tune):
    
    """
    
    naive bayes algorithm
    
    """
    
    class_X_train = pd.DataFrame(X_train['class'].value_counts(normalize = True))

    class_X_train = class_X_train.rename(columns = {'class' : 'class probability'}).rename_axis(['class'])

    # use the groupby function to calculate the conditional probability for each attribute given the class information
    final_X_train = pd.DataFrame()
    for col in X_train.iloc[:, :-1].columns:
        output_X_train = pd.DataFrame(X_train.groupby(['class', col])[col].count().groupby(level = 0).apply(lambda x:  (x + m * p) / float(x.sum() + m))).rename_axis(['class', 'label'])    
        final_X_train = pd.concat([final_X_train, output_X_train], axis = 1)  
        final_X_train = final_X_train.fillna(0)

    prob_X_train = pd.merge(final_X_train, class_X_train,  left_index = True, right_index = True)

    prob_X_train = prob_X_train.reset_index(drop = False)
    
    
    # find the conditional probability using the data instance's class label
    # calculate the conditional probability for each class given the attribute
    total_prob = {}
    for row in range(0, X_tune.shape[0]):
        prob_class = {}
        for c in range(0, len(X_tune['class'].unique())):
            prob_list = []
            for key, items in X_tune.iloc[row, :-1].to_dict().items():
                prob_list.append(prob_X_train[prob_X_train['class'] == c][(prob_X_train['label'] == items)][key].values)
            prob_class[c] = np.prod(prob_list) * prob_X_train[prob_X_train['class'] == c]['class probability'].values[0]
        total_prob[row] = prob_class

    final_prob = pd.DataFrame(total_prob).T
    
    X_tune = X_tune.reset_index(drop = True)

    X_tune['Prediction'] = final_prob.eq(final_prob.max(1), axis = 0).dot(final_prob.columns)
    
    accuracy = X_tune[X_tune['class'] == X_tune['Prediction']].shape[0] / X_tune.shape[0]
    X_tune = X_tune.drop('Prediction', axis = 1)
    
    para_list = [m, p, accuracy]
    
    return para_list, prob_X_train


# In[80]:


def naive_bayes_parameters(X_train, X_tune):
    """
    return the accuracy of the Naive Bayes Algorithm using different m and p
    """
    m_list = [1, 10, 20]

    p_list = [0.001, 0.01, 0.5]
    
    output_list = []

    for m in m_list:
        for p in p_list:
            para_list, prob_X_train = naive_bayes(m, p, X_train, X_tune)
            output_list.append(para_list)
            
    parameters_df = pd.DataFrame(output_list, columns = ['m', 'p', 'Accuracy'])
    
    return parameters_df, prob_X_train


# In[81]:


def optimal_parameter(X_train, X_test, output_df):
    """
    find the best hyperparameters after tuning and return the accuracy on the test set
    """
    optimal_m = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'm']

    optimal_p = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'p']
    
    X_train = X_train.reset_index(drop = True)
    
    return naive_bayes(optimal_m, optimal_p, X_train, X_test)


# In[82]:


# Breast Cancer Dataset:

X_tune, X_train, X_test = train_tune_test(one_hot_breast)

output_df, prob_X_train_tune = naive_bayes_parameters(X_train, X_tune)


# In[83]:


X_tune


# In[84]:


X_train


# In[85]:


X_test


# In[86]:


output_df


# In[87]:


prob_X_train_tune


# In[88]:


para_list, prob_X_train = optimal_parameter(X_train, X_test, output_df)


# In[89]:


para_list


# In[90]:


prob_X_train


# In[91]:


# Glass Dataset:

X_tune, X_train, X_test = train_tune_test(final_glass)

output_df, prob_X_train_tune = naive_bayes_parameters(X_train, X_tune)

output_df


# In[92]:


para_list, prob_X_train = optimal_parameter(X_train, X_test, output_df)


# In[93]:


para_list


# In[94]:


# Iris Dataset:

X_tune, X_train, X_test = train_tune_test(one_hot_iris)

output_df, prob_X_train_tune = naive_bayes_parameters(X_train, X_tune)

output_df


# In[95]:


para_list, prob_X_train = optimal_parameter(X_train, X_test, output_df)


# In[96]:


para_list


# In[97]:


# Soybean Dataset:

X_tune, X_train, X_test = train_tune_test(final_soybean)

output_df, prob_X_train_tune = naive_bayes_parameters(X_train, X_tune)

output_df


# In[98]:


para_list, prob_X_train = optimal_parameter(X_train, X_test, output_df)


# In[99]:


para_list


# In[100]:


# Vote Dataset:

X_tune, X_train, X_test = train_tune_test(final_vote)

output_df, prob_X_train_tune = naive_bayes_parameters(X_train, X_tune)

output_df


# In[101]:


para_list, prob_X_train = optimal_parameter(X_train, X_test, output_df)


# In[102]:


para_list


# In[ ]:





# In[ ]:




