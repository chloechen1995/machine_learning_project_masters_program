
# Logistic Regression

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle


# Breast Cancer Dataset:
# 
# Glass Dataset:
# 
# Iris Dataset:
# 
# Soybean Dataset:
# 
# Vote Dataset:
# 

# Breast Cancer Dataset:
# 
# • Missing values handling: separate the observations with missing values based on their classes and replace the missing values with the average of the attribute in the class (Mean Imputation)

# – Provide sample outputs from one test set showing classification performance on Adaline and
# Logistic Regression.
# 
# – Show a sample trained Adaline model and Logistic Regression model.
# 
# – Demonstrate the weight updates for Adaline and Logistic Regression. For Logistic Regression,
# show the multi-class case.
# 
# – Demonstrate the gradient calculation for Adaline and Logistic Regression. For Logistic Regression,
# show the multi-class case.
# 
# – Show the average performance over the five folds for Adaline and Logistic Regression.

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


def normalize_attr(df):
    """
    this function normalizes the numerical attributes so that it ranges from -1 to +1
    """
    attr_cols = df.loc[:, df.columns != 'class'].columns
    attr_df = pd.DataFrame(preprocessing.scale(df.loc[:, df.columns != 'class']), columns = attr_cols)
    final_df = pd.concat([attr_df, df['class']], axis = 1)
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


breast['class'] = breast['class'].astype('category').cat.codes


# In[8]:


breast_df = fillnan(breast)


# In[9]:


breast_df = breast_df.drop(['Sample code number'], axis = 1)


# In[10]:


breast_df = normalize_attr(breast_df)


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


glass_df = normalize_attr(glass)


# In[16]:


glass_df.head()


# Iris Dataset:

# In[17]:


iris = pd.read_csv('iris.data', sep = ',', header = None)


# In[18]:


iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']


# In[19]:


iris['class'] = iris['class'].astype('category').cat.codes


# In[20]:


iris_df = normalize_attr(iris)


# In[21]:


iris_df.head()


# Soybean Dataset:
# 
# • No missing values

# In[22]:


soybean = pd.read_csv('soybean-small.data', sep = ',', header = None)


# In[23]:


soybean = soybean.rename(columns = {35 : 'class'}) 


# In[24]:


soybean['class'] = soybean['class'].astype('category').cat.codes


# In[25]:


soybean_df = normalize_attr(soybean)


# In[26]:


soybean_df.head()


# Vote Dataset:
# 
# • Replace all ‘y’ with 1 and ‘n’ with 0 
# 
# • Missing values handling: separate the observations with missing values based on their classes and replace the missing values with the average of the attribute in the class

# In[27]:


vote = pd.read_csv('house-votes-84.data', sep = ',', header = None)


# In[28]:


vote.columns = ['class', 'handicapped-infants', 'water-project-cost-sharing',
               'adoption-of-the-budget-resolution', 'physician-fee-freeze', 
               'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
               'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
               'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
               'export-administration-act-south-africa']


# In[29]:


vote = vote.replace('?', np.nan)


# In[30]:


vote = vote.replace('n', 0)


# In[31]:


vote = vote.replace('y', 1)


# In[32]:


vote['class'] = vote['class'].astype('category').cat.codes


# In[33]:


vote_df = fillnan(vote)


# In[34]:


vote_df.head()


# In[35]:


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


# In[36]:


def sigmoid(x):
    """
    sigmoid function
    """
    return 1/(1+np.exp(-x))


# In[37]:


def gradient_descent(X_train, lr, max_iterations, threshold):
    """
    compute the gradient descent
    """
    
    X_attribute = X_train.loc[:, ((X_train.columns != 'class') & (X_train.columns != 'class_multiple'))]
    num_instances, num_attributes = X_attribute.shape
    
    wj = np.random.uniform(-0.01, 0.01, num_attributes)
    
    weight_change_boolean = True
    
    i = 0
    while weight_change_boolean and i <= max_iterations:
        change_wj = np.zeros(num_attributes)

        # for each training instance
        for row in range(0, num_instances):
            # calculate the weighted sum
            weighted_sum = np.dot(wj, X_attribute.loc[row, :])
            # predicted value
            predicted = sigmoid(weighted_sum)
            #print(predicted)
            # actual value
            actual = X_train.loc[row, 'class']
            #print(actual)
            # gradient
            gradient = (actual - predicted) * X_attribute.loc[row, :]
            #print(gradient)
            change_wj = change_wj + gradient

        # calculate the step size
        step_size = change_wj * lr
        # update the weight vectors
        wj = wj + step_size

        # check whether the change in weight vector is minimal, if yes, we stop the algorithm

        improvement = sum(abs(change_wj)**2)**(1/2)
        #print(improvement)

        if improvement < threshold:
            weight_change_boolean = False

        i+= 1
        
    return wj
    


# In[38]:


def logistic_regression(weights, X_tune):
    
    """
    apply logistic regression on the unseen dataset and obtains the accuracy
    """
    
    X_attribute = X_tune.loc[:, X_tune.columns != 'class']
    num_instances, num_attributes = X_attribute.shape
    
    accurate_count = 0

    for row in range(0, num_instances):
        # calculate the weighted sum
        weighted_sum = np.dot(weights, X_attribute.loc[row, :])
        # predicted value
        predicted = sigmoid(weighted_sum)
        #print('predicted', predicted)
        # actual value
        actual = X_tune.loc[row, 'class']
        #print('actual', actual)
        if predicted >= 0.5:
            predicted_val = 1
        else:
            predicted_val = 0

        if predicted_val == actual:
            accurate_count += 1
            
    return accurate_count / num_instances
    


# In[39]:


def train_tune_test_multi_class(df):
    
    """
    use the original dataset to create training set, test set and tuning set for multiclass
    """
    
    df = df.rename(columns = {'class': 'class_multiple'})
    return df


# In[40]:


def logistic_multiclass_training(X_train, lr, max_iterations, threshold):
    """
    logistic regression training for multiclass dataset
    """    
    weights_train_vector = {}
    
    # create one classifier for each class in the dataset
    for i in range(0, len(X_train['class_multiple'].unique())):
        X_train['class'] = np.where(X_train['class_multiple'] == i, 1, 0)
        weights_train = gradient_descent(X_train, lr, max_iterations, threshold)
        weights_train_vector[i] = weights_train
    # train the model for each class and get the weights vector, which results in K weights vectors assuming we have K classes
    return weights_train_vector


# In[41]:


def logistic_multiclass_test(X_test, weights_train_vector):
    """
    logistic regression testing algorithm for multiclass dataset
    """
    
    X_attribute = X_test.loc[:, ((X_test.columns != 'class') & (X_test.columns != 'class_multiple') & ((X_test.columns != 'Prediction')))]
    #print(X_attribute)
    weight_list_df = []
    for key, value in weights_train_vector.items():
        weight_list = []
        for row in range(0, X_attribute.shape[0]):
            weighted_sum = np.dot(np.array(X_attribute.iloc[row, :]), weights_train_vector[key])
            weight_list.append(weighted_sum)
        weight_list_df.append(weight_list)

    weight_df = pd.DataFrame(weight_list_df).T
    # the observation is classified as the class with the highest weighted sum
    X_test['Prediction']= pd.DataFrame(weight_df.eq(weight_df.max(1), axis = 0).dot(weight_df.columns), columns = ['Prediction'])
    accuracy = X_test[X_test['class_multiple'] == X_test['Prediction']].shape[0] / X_test.shape[0]
    return accuracy


# In[42]:


def logistic_regression_training(X_tune, X_train, class_type):
    """
    Hyperparameter tuning process 
    """
    lr_list = [0.01, 0.05, 0.1]
    max_iterations_list = [10, 20, 30]
    threshold_list = [0.001, 0.005, 0.1]
    
    output_list = []
    
    for lr in lr_list:
        for max_iterations in max_iterations_list:
            for threshold in threshold_list:
                if class_type == 'binary':
                    weights = gradient_descent(X_train, lr, max_iterations, threshold)
                    output_list.append([lr, max_iterations, threshold, weights, logistic_regression(weights, X_tune)])
                elif class_type == 'multiclass':
                    weights_train_vector = logistic_multiclass_training(X_train, lr, max_iterations, threshold)
                    output_list.append([lr, max_iterations, threshold, weights_train_vector, logistic_multiclass_test(X_tune, weights_train_vector)])

    output_df = pd.DataFrame(output_list, columns = ['Learning Rate', 'Max Iterations', 'Threshold', 'Weights', 'Accuracy'])
    
    return output_df
                
                


# In[43]:


def optimal_parameter(output_df):
    """
    find the best hyperparameters after tuning and return its accuracy on the test set
    """
    
    optimal_lr = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'Learning Rate']
    
    optimal_iterations = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'Max Iterations']
    
    optimal_threshold = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'Threshold']
    
    return optimal_lr, optimal_iterations, optimal_threshold


# In[44]:


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


# In[45]:


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


# In[46]:


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


# In[47]:


def logistic_regression_cross_validation(data_df, class_type):
    
    data_df['bias'] = 1
    
    if class_type == 'binary':
        tuning_set, remaining_set = tune_remaining(data_df, 'class')
        data_subset_dict = create_fold_classification(remaining_set, 'class')
    elif class_type == 'multiclass':
        data_df = train_tune_test_multi_class(data_df)
        tuning_set, remaining_set = tune_remaining(data_df, 'class_multiple')
        data_subset_dict = create_fold_classification(remaining_set, 'class_multiple')

    test_result = []
    for num in range(1, 6):
        training_set, test_set = train_test_c(data_subset_dict, num)
        
        if class_type == 'binary':
            output_df = logistic_regression_training(tuning_set, training_set, 'binary')
            optimal_lr, optimal_iterations, optimal_threshold = optimal_parameter(output_df)
            print('Optimal Learning Rate', optimal_lr)
            print('Optiaml Iterations', optimal_iterations)
            print('Optiamal Threshold', optimal_threshold)
            
            weights = gradient_descent(training_set, optimal_lr, optimal_iterations, optimal_threshold)
            accuracy = logistic_regression(weights, test_set)
            print("fold {}'s accuracy is {}".format(num, accuracy))
            test_result.append(accuracy)
        elif class_type == 'multiclass':
            output_df = logistic_regression_training(tuning_set, training_set, 'multiclass')
            optimal_lr, optimal_iterations, optimal_threshold = optimal_parameter(output_df)
            print('Optimal Learning Rate', optimal_lr)
            print('Optiaml Iterations', optimal_iterations)
            print('Optiamal Threshold', optimal_threshold)
            
            weights_train_vector = logistic_multiclass_training(training_set, optimal_lr, optimal_iterations, optimal_threshold)
            accuracy = logistic_multiclass_test(test_set, weights_train_vector)
            print("fold {}'s accuracy is {}".format(num, accuracy))
            test_result.append(accuracy)

    test_average = sum(test_result) / len(test_result)
    
    return test_average
    


# In[48]:


# breast dataset
logistic_regression_cross_validation(breast_df, 'binary')


# In[ ]:


# glass dataset
logistic_regression_cross_validation(glass_df, 'multiclass')


# In[ ]:


# iris dataset
logistic_regression_cross_validation(iris_df, 'multiclass')


# In[ ]:


# soybean dataset
logistic_regression_cross_validation(soybean_df, 'multiclass')


# In[ ]:


# vote dataset
logistic_regression_cross_validation(vote_df, 'binary')


# In[ ]:


hello


# In[ ]:


data_df['bias'] = 1
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)

test_result = []
for num in range(1, 6):
    training_set, test_set = train_test_c(data_subset_dict, num)

    output_df = logistic_regression_training(tuning_set, training_set, 'binary')
    optimal_lr, optimal_iterations, optimal_threshold = optimal_parameter(output_df)
    
    test_result.append(logistic_regression_cross_validation(tuning_set, training_set, test_set, 'binary'))
test_average = sum(test_result) / len(test_result)


# In[ ]:





# – Provide sample outputs from one test set showing classification performance on Adaline and
# Logistic Regression.

# In[ ]:


data_df = breast_df
data_df['bias'] = 1
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)
num = 1
training_set, test_set = train_test_c(data_subset_dict, num)
# breast dataset
logistic_regression_cross_validation(tuning_set, training_set, test_set, 'binary')


# glass dataset
logistic_regression_main(glass_df, 'multiclass')


# In[ ]:


# iris dataset
logistic_regression_main(iris_df, 'multiclass')


# In[ ]:


# soybean dataset
logistic_regression_main(soybean_df, 'multiclass')


# In[ ]:


# vote dataset
logistic_regression_main(vote_df, 'binary')



# Adaline

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle


# Breast Cancer Dataset:
# 
# Glass Dataset:
# 
# Iris Dataset:
# 
# Soybean Dataset:
# 
# Vote Dataset:
# 

# Breast Cancer Dataset:
# 
# • Missing values handling: separate the observations with missing values based on their classes and replace the missing values with the average of the attribute in the class (Mean Imputation)

# – Provide sample outputs from one test set showing classification performance on Adaline and
# Logistic Regression.
# 
# – Show a sample trained Adaline model and Logistic Regression model.
# 
# – Demonstrate the weight updates for Adaline and Logistic Regression. For Logistic Regression,
# show the multi-class case.
# 
# – Demonstrate the gradient calculation for Adaline and Logistic Regression. For Logistic Regression,
# show the multi-class case.
# 
# – Show the average performance over the five folds for Adaline and Logistic Regression.

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


def normalize_attr(df):
    """
    this function normalizes the numerical attributes so that it ranges from -1 to +1
    """
    attr_cols = df.loc[:, df.columns != 'class'].columns
    attr_df = pd.DataFrame(preprocessing.scale(df.loc[:, df.columns != 'class']), columns = attr_cols)
    final_df = pd.concat([attr_df, df['class']], axis = 1)
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


breast['class'] = breast['class'].map({2: -1, 4:1})


# In[8]:


breast_df = fillnan(breast)


# In[9]:


breast_df = breast_df.drop(['Sample code number'], axis = 1)


# In[10]:


breast_df = normalize_attr(breast_df)


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


glass_df = normalize_attr(glass)


# In[16]:


glass_df.head()


# Iris Dataset:

# In[17]:


iris = pd.read_csv('iris.data', sep = ',', header = None)


# In[18]:


iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']


# In[19]:


iris['class'] = iris['class'].astype('category').cat.codes


# In[20]:


iris_df = normalize_attr(iris)


# In[21]:


iris_df.head()


# Soybean Dataset:
# 
# • No missing values

# In[22]:


soybean = pd.read_csv('soybean-small.data', sep = ',', header = None)


# In[23]:


soybean = soybean.rename(columns = {35 : 'class'}) 


# In[24]:


soybean['class'] = soybean['class'].astype('category').cat.codes


# In[25]:


soybean_df = normalize_attr(soybean)


# In[26]:


soybean_df.head()


# Vote Dataset:
# 
# • Replace all ‘y’ with 1 and ‘n’ with 0 
# 
# • Missing values handling: separate the observations with missing values based on their classes and replace the missing values with the average of the attribute in the class

# In[27]:


vote = pd.read_csv('house-votes-84.data', sep = ',', header = None)


# In[28]:


vote.columns = ['class', 'handicapped-infants', 'water-project-cost-sharing',
               'adoption-of-the-budget-resolution', 'physician-fee-freeze', 
               'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
               'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
               'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
               'export-administration-act-south-africa']


# In[29]:


vote = vote.replace('?', np.nan)


# In[30]:


vote = vote.replace('n', 0)


# In[31]:


vote = vote.replace('y', 1)


# In[32]:


vote['class'] = vote['class'].map({'republican': -1, 'democrat':1})


# In[33]:


vote_df = fillnan(vote)


# In[34]:


vote_df.head()


# In[80]:


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


# In[81]:


def weight_update(df, weights_vector, lr):
    total_error = 0
    for i in range(len(df)):
        #print('weight_vector', weights_vector)
        X = np.array(df.loc[i, df.columns != 'class'].tolist())
        X_input = np.insert(X, 0, 1)
        #print('Input value', X_input)
        predicted = float(np.dot(X_input, weights_vector))
        #print('predicted value: ', predicted)
        y = df.loc[i, 'class']
        #print('actual class', y)
        #error = (y - predicted)**2
        #total_error += error
        #print('error:', error)
        updated_weights = weights_vector + lr * (y - predicted)  * X_input
        #print('updated weights:', updated_weights)
        w_change = updated_weights - weights_vector
        #print('weight_change:', w_change)
        weights_vector = updated_weights
    return weights_vector


# In[82]:


def adaline_training(X_train, num_epoch, lr):
    """
    adaline training algorithm
    """
    X_attribute = X_train.loc[:, (X_train.columns != 'class') & (X_train.columns != 'class_multiple')]
    
    num_instances, num_attributes = X_attribute.shape
    
    weights_vector = np.random.uniform(-0.01, 0.01, num_attributes + 1)
    
    for i in range(num_epoch):
        X_train_input = X_train.loc[:, (X_train.columns != 'class_multiple')]
        weights_vector = weight_update(X_train_input, weights_vector, lr)
        
    return weights_vector


# In[83]:


def adaline_test(X_test, weights_vector):
    """
    apply adaline to unseen dataset
    """
    a_count = 0
    for i in range(len(X_test)):
        X = np.array(X_test.loc[i, ((X_test.columns != 'class') & (X_test.columns != 'class_multiple') & ((X_test.columns != 'Prediction')))].tolist())
        X_input = np.insert(X, 0, 1)
        predicted = float(np.dot(X_input, weights_vector))
        if predicted < 0:
            predicted_val = -1
        else:
            predicted_val = 1

        y = X_test.loc[i, 'class']

        if predicted_val == y:
            a_count += 1
    return a_count/len(X_test)
    


# In[84]:


def train_tune_test_multi_class(df):
    
    """
    use the original dataset to create training set, test set and tuning set for multiclass
    """
    
    df = df.rename(columns = {'class': 'class_multiple'})
    return df


# In[85]:


def adaline_multiclass_training(X_train, num_epoch, lr):
    """
    adaline training for multiclass dataset
    """    
    weights_train_vector = {}
    
    # create one classifier for each class in the dataset
    for i in range(0, len(X_train['class_multiple'].unique())):
        X_train['class'] = np.where(X_train['class_multiple'] == i, 1, 0)
        weights_train = adaline_training(X_train, num_epoch, lr)
        weights_train_vector[i] = weights_train
    # train the model for each class and get the weights vector, which results in K weights vectors assuming we have K classes
    return weights_train_vector


# In[86]:


def adaline_multiclass_test(X_test, weights_train_vector):
    """
    logistic regression testing algorithm for multiclass dataset
    """
    
    X_attribute = X_test.loc[:, ((X_test.columns != 'class') & (X_test.columns != 'class_multiple') & ((X_test.columns != 'Prediction')))]
    weight_list_df = []
    for key, value in weights_train_vector.items():
        weight_list = []
        for row in range(0, X_attribute.shape[0]):
            X = X_attribute.iloc[row, :].tolist()
            X_input = np.insert(X, 0, 1)
            weighted_sum = np.dot(np.array(X_input), weights_train_vector[key])
            weight_list.append(weighted_sum)
        weight_list_df.append(weight_list)

    weight_df = pd.DataFrame(weight_list_df).T
    # the observation is classified as the class with the highest weighted sum
    X_test['Prediction']= pd.DataFrame(weight_df.eq(weight_df.max(1), axis = 0).dot(weight_df.columns), columns = ['Prediction'])
    accuracy = X_test[X_test['class_multiple'] == X_test['Prediction']].shape[0] / X_test.shape[0]
    return accuracy


# In[87]:


def adaline_training_parameter(X_tune, X_train, class_type):
    
    """
    Hyperparameter tuning process
    """
    lr_list = [0.01, 0.001, 0.0001]
    num_epochs = [10, 30, 50]
    output_list = []

    for lr in lr_list:
        for num_epoch in num_epochs:
            if class_type == 'binary':
                weights_vector = adaline_training(X_train, num_epoch, lr)
                output_list.append([lr, num_epoch, weights_vector, adaline_test(X_tune, weights_vector)])
            elif class_type == 'multiclass':
                weights_trained_vector = adaline_multiclass_training(X_train, num_epoch, lr)
                output_list.append([lr, num_epoch, weights_trained_vector, adaline_multiclass_test(X_tune, weights_trained_vector)])

    output_df = pd.DataFrame(output_list, columns = ['Learning Rate', '# Epochs', 'Weights', 'Accuracy'])
    
    return output_df


# In[88]:


def optimal_parameter(output_df):
    """
    find the best hyperparameters after tuning and return its accuracy on the test set
    """
    
    optimal_lr = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, 'Learning Rate']
    
    optimal_epoch = output_df[output_df['Accuracy'] == output_df['Accuracy'].max()].reset_index().loc[0, '# Epochs']
    

    return optimal_lr, optimal_epoch
    


# In[89]:


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


# In[90]:


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


# In[91]:


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


# In[92]:


def adaline_cross_validation(data_df, class_type):
    """
    cross validation algorithm for adaline
    """
    if class_type == 'binary':
        tuning_set, remaining_set = tune_remaining(data_df, 'class')
        data_subset_dict = create_fold_classification(remaining_set, 'class')
    elif class_type == 'multiclass':
        data_df = train_tune_test_multi_class(data_df)
        tuning_set, remaining_set = tune_remaining(data_df, 'class_multiple')
        data_subset_dict = create_fold_classification(remaining_set, 'class_multiple')
        
    test_result = []
    for num in range(1, 6):
        training_set, test_set = train_test_c(data_subset_dict, num)
    
        if class_type == 'binary':
            output_df = adaline_training_parameter(tuning_set, training_set, 'binary')
            optimal_lr, optimal_epoch = optimal_parameter(output_df)
            print('Optimal Learning Rate', optimal_lr)
            print('Optimal # of Epoch', optimal_epoch)
            weights_vector = adaline_training(training_set, optimal_epoch, optimal_lr)
            accuracy = adaline_test(test_set, weights_vector)
            print("fold {}'s accuracy is {}".format(num, accuracy))
            test_result.append(accuracy)

        elif class_type == 'multiclass':
            output_df = adaline_training_parameter(tuning_set, training_set, 'multiclass')
            optimal_lr, optimal_epoch = optimal_parameter(output_df)
            print('Optimal Learning Rate', optimal_lr)
            print('Optimal # of Epoch', optimal_epoch)
            weights_trained_vector = adaline_multiclass_training(training_set, optimal_epoch, optimal_lr)
            accuracy = adaline_multiclass_test(test_set, weights_trained_vector)
            print("fold {}'s accuracy is {}".format(num, accuracy))
            test_result.append(accuracy)

        test_average = sum(test_result) / len(test_result)
    
    return test_average
        


# In[48]:


# breast dataset
# optimal_lr, optimal_epoch, accuracy
adaline_cross_validation(breast_df, 'binary')


# In[93]:


# glass dataset
adaline_cross_validation(glass_df, 'multiclass')


# In[94]:


# iris dataset
adaline_cross_validation(iris_df, 'multiclass')


# In[95]:


# soybean dataset
adaline_cross_validation(soybean_df, 'multiclass')


# In[96]:


# vote dataset
adaline_cross_validation(vote_df, 'binary')