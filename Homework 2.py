#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import random
random.seed(4)
import seaborn as sns
import math
import re


# Classification Datasets:
# * Glass
# * Image Segmentation
# * Vote
# 
# Regression Datasets:
# * Abalone
# * Computer Hardware
# * Forest Fires

# Preprocess the datasets
# 
# Glass Dataset:
# * Remove the id number column because it is unique to each observation
#     
# Segmentation Dataset:
# * Assign numerical values to the class
# * Remove 'REGION-PIXEL-COUNT' column because they are the same
# 
# Vote Dataset:
# * No changes needed
# 
# Abalone Dataset:
# * Assign numerical values to the sex category
# 
# Computer Hardware Dataset:
# * Remove the 'Model Name' column because it is unique to each observation
# * Normalize the attributes so that they are between 0 and 1
# 
# Forest Fires Dataset
# * Assign numerical values to month and day
# 
# 

# #### Glass Dataset



glass = pd.read_csv('glass.data', sep = ',', header = None)

glass.columns = ['Id number', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']


glass = glass.drop(['Id number'], axis = 1)

scaler = MinMaxScaler()



glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']] = scaler.fit_transform(glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']])


glass


# #### Segmentation Dataset


seg_data = []
with open('segmentation.data') as segmentation:
    for f in segmentation:
        seg_data.append(f)
seg_df = pd.DataFrame(seg_data)
seg_final_df = seg_df[5:][0].str.split(',', expand = True)


seg_list = ['class']
for i in seg_df.loc[3].str.split(','):
    seg_list.extend(i)

seg_final_df.columns = seg_list

seg_final_df = seg_final_df.rename(columns = {'HUE-MEAN\n': 'HUE-MEAN'})

seg_final_df['HUE-MEAN'] = seg_final_df['HUE-MEAN'].str.rstrip()


seg_final_df = seg_final_df.reset_index(drop = True)


seg_final_df['class'] = pd.factorize(seg_final_df['class'])[0]

seg_final_df = seg_final_df.drop(['REGION-PIXEL-COUNT'], axis = 1)


seg_final_df



seg_final_df[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'INTENSITY-MEAN', 'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN', 'EXGREEN-MEAN', 'VALUE-MEAN', 'HUE-MEAN']] = scaler.fit_transform(seg_final_df[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'INTENSITY-MEAN', 'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN', 'EXGREEN-MEAN', 'VALUE-MEAN', 'HUE-MEAN']])



seg_final_df.loc[:, seg_final_df.columns != 'class'] = seg_final_df.loc[:, seg_final_df.columns != 'class'].astype(float)


# #### Vote Dataset


vote = pd.read_csv('house-votes-84.data', sep = ',', header = None)


vote.columns = ['class', 'handicapped-infants', 'water-project-cost-sharing',
               'adoption-of-the-budget-resolution', 'physician-fee-freeze', 
               'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
               'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
               'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
               'export-administration-act-south-africa']


vote['class'] = vote['class'].astype('category').cat.codes



vote


# #### Abalone Dataset



abalone = pd.read_csv('abalone.data', sep = ',', header = None)


abalone.head()


abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'target']



abalone['Sex'] = pd.factorize(abalone['Sex'])[0]



abalone.shape


# #### Computer Hardware Dataset

machine = pd.read_csv('machine.data', header = None)


machine.head()



machine.columns = ['Vendor name', 'Model Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'target', 'ERP']


machine = machine.drop(['Model Name', 'ERP'], axis = 1)


machine['Vendor name'] = pd.factorize(machine['Vendor name'])[0]


machine[['Vendor name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']] = scaler.fit_transform(machine[['Vendor name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']])


machine.shape


# #### Forestfire Dataset


forestfires = pd.read_csv('forestfires.data')


forestfires = forestfires.rename(columns = {'area': 'target'})


forestfires['month'] = pd.factorize(forestfires['month'])[0] 


forestfires['day'] = pd.factorize(forestfires['day'])[0]


forestfires.shape


# – Show your data being split into five folds for one of the data sets.
# 
# – Demonstrate the calculation of your distance function.
# 
# – Demonstrate the calculation of your kernel function.
# 
# – Demonstrate an example of a point being classified using k-nn. Show the neighbors returned as
# well as the point being classified.
# 
# – Demonstrate an example of a point being regressed using k-nn. Show the neighbors returned as
# well as the point being predicted.
# 
# – Demonstrate an example being edited out of the training set using edited nearest neighbor.
# 
# – Demonstrate an example being added to the training set using condensed nearest neighbor.
# 
# – Show the average performance across the five folds for each of k-nn, ENN, and CNN on a classification data set.
# 
# – Show the average performance across the five folds for k-nn on a regression data set.

# ### Create training set, test set and tuning set


def tune_remaining(data_df):
    """
    take out 10% of the dataset for tuning and 90% for training and testing
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


# ### Implement k-nearest neighbor


# knn classification with numeric variables


def euclidean_distance(training_set_np, test_set_np):
    """
    calculate the euclidean distance between the training set and the observation in the test set
    """
    return pd.DataFrame(np.sqrt(np.sum(np.square(test_set_np - training_set_np), axis = 1)), columns = ['sum_difference'])



def knn_classification_numeric(training_set, test_set, k):
    """
    knn classifier for classification dataset
    """
    test_set['predicted_class'] = np.nan
    for i in range(0, len(test_set)):
        training_set_np = np.array(training_set.loc[:, training_set.columns != 'class']).astype(float)
        test_set_np = np.array(test_set.drop(['class', 'predicted_class'], axis = 1).loc[i, :]).astype(float)
        sum_sqaured = euclidean_distance(training_set_np, test_set_np)
        label_df = training_set['class']

        # find the k neighbors based on the sum of squared distance
        neighbor_df = pd.DataFrame(pd.concat([label_df, sum_sqaured], axis = 1).nsmallest(k, 'sum_difference').groupby(['class'])['class'].count()).rename(columns = {'class': '# observations'}).reset_index()

        test_set.loc[i, 'predicted_class'] = neighbor_df.sort_values(by = '# observations', ascending = False).reset_index(drop = True).loc[0, 'class']

    
    return len(test_set[test_set['predicted_class'] != test_set['class']]) / len(test_set['class'])



def k_classifier_numeric(training_set, tuning_set):
    """
    calculate the classification error for each k value
    """
    
    k_accuracy = []
    
    for k in range(1,10):
        k_accuracy.append([k, knn_classification_numeric(training_set, tuning_set, k)])
        
    final_k_df = pd.DataFrame(k_accuracy, columns = ['K', 'Classification Error'])
    
    return final_k_df




def cross_validation_c_n(k, data_subset_dict):
    """
    do five fold cross validation with the training and test set with the optimal k
    """
    for num in range(1, 6):
        test_result = []
        training_set, test_set = train_test_c(data_subset_dict, num)
        test_result.append(knn_classification_numeric(training_set, test_set, k))
        test_average = sum(test_result) / len(test_result)
    
    return test_average
        
    


def visual_k(knn_class, metric, dname):
    """
    visualize the elbow plot
    """
    return sns.lineplot(x = knn_class['K'], y = knn_class[metric]).set_title('k-NN Algorithm Classification Error For Various K - ' + dname)





data_df = glass
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)



num = 1
training_set, test_set = train_test_c(data_subset_dict, num)



knn_class = k_classifier_numeric(training_set, tuning_set)




knn_class

visual_k(knn_class, 'Classification Error', 'Glass')


optimal_k = knn_class[knn_class['Classification Error'] == knn_class['Classification Error'].min()]['K'].values[0]


optimal_k


knn_glass = cross_validation_c_n(optimal_k, data_subset_dict) 


knn_glass




data_df = seg_final_df
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)



training_set, test_set = train_test_c(data_subset_dict, num)



knn_class = k_classifier_numeric(training_set, tuning_set)


knn_class



visual_k(knn_class, 'Classification Error', 'Segmentation')



optimal_k = knn_class[knn_class['Classification Error'] == knn_class['Classification Error'].min()]['K'].values[0]



optimal_k


knn_segmentation = cross_validation_c_n(optimal_k, data_subset_dict) 



knn_segmentation


def distance_df(col, data_df):
    """
    create a dataframe to store the distance between the categorical variables for an attribute
    """
    col_df = pd.DataFrame(data_df.groupby([col])[col].count()).rename(columns = {col: col + '_count_total'}).reset_index()
    c_col_df = pd.DataFrame(data_df.groupby(['class', col])[col].count()).rename(columns = {col: col + '_count'}).reset_index()
    f_df = pd.merge(c_col_df, col_df, on = col)
    f_df['percentage'] = f_df[col + '_count'] / f_df[col + '_count_total']
    
    total_dict = {}
    for c in f_df['class'].unique():
        class_dict = {}
        for i in f_df[col].unique():
            cia = np.array(f_df[(f_df['class'] == c) & (f_df[col] == i)]['percentage'])
            cjas = np.array(f_df[(f_df['class'] == c)]['percentage'])
            class_dict[i] = np.abs(cia - cjas)**2
        total_dict[c] = class_dict
    
    d_df = pd.DataFrame.from_dict(total_dict)
    d_df['prob'] = d_df.sum(axis = 1)
    
    output = pd.DataFrame(d_df['prob'].tolist())
    
    output.index = d_df.index
    
    output.columns = d_df.index
    
    return output
    

data_df = vote
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)


training_set, test_set = train_test_c(data_subset_dict, num)

dict_df = {}

for c in data_df.loc[:, data_df.columns != 'class'].columns.unique():
    dict_df[c] = distance_df(c, data_df)




def knn_classification_categorical(training_set, test_set, k):
    """
    knn classification for categorical variable
    """
    test_set['predicted_class'] = np.nan
    for i in range(0, len(test_set)):
        total_distance = {}
        label_df = training_set['class']
        for j in range(len(training_set)):
            distance = []
            for col in training_set.loc[:, training_set.columns != 'class'].columns.unique():
                train_val = training_set.loc[j, col]
                test_val = test_set.loc[i, col]
                distance.append(dict_df[col][train_val][test_val])
                total_distance[j] = np.sqrt(np.sum(distance))
        sum_squared = pd.DataFrame.from_dict([total_distance]).T
        sum_squared.columns = ['sum_difference']
        
            # find the k neighbors based on the sum of squared distance
        neighbor_df = pd.DataFrame(pd.concat([label_df, sum_squared], axis = 1).nsmallest(k, 'sum_difference').groupby(['class'])['class'].count()).rename(columns = {'class': '# observations'}).reset_index()
        test_set.loc[i, 'predicted_class'] = neighbor_df.sort_values(by = '# observations', ascending = False).reset_index(drop = True).loc[0, 'class']

    return len(test_set[test_set['predicted_class'] != test_set['class']]) / len(test_set['class'])



def k_classifier_cat(training_set, tuning_set):
    """
    calculate the classification error for each k value
    """
    
    k_accuracy = []
    
    for k in range(1,10):
        k_accuracy.append([k, knn_classification_categorical(training_set, tuning_set, k)])
        
    final_k_df = pd.DataFrame(k_accuracy, columns = ['K', 'Classification Error'])
    
    return final_k_df


def cross_validation_c_c(k, data_subset_dict):
    """
    do five fold cross validation with the training and test set with the optimal k
    """
    for num in range(1, 6):
        test_result = []
        training_set, test_set = train_test_c(data_subset_dict, num)
        test_result.append(knn_classification_categorical(training_set, test_set, k))
        test_average = sum(test_result) / len(test_result)
    
    return test_average
        



knn_class = k_classifier_cat(training_set, tuning_set)

knn_class


visual_k(knn_class, 'Classification Error', 'Vote')



optimal_k = knn_class[knn_class['Classification Error'] == knn_class['Classification Error'].min()]['K'].values[0]


optimal_k


knn_vote = cross_validation_c_c(optimal_k, data_subset_dict)



knn_vote


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



def gaussian_kernel(neighbor_df, sigma):
    """
    use the Gaussian kernel to make prediction on the query point
    """
    neighbor_df['kernel'] = np.exp(1/(2*sigma) * neighbor_df['sum_difference'])
    neighbor_df['numerator'] = neighbor_df['kernel'] * neighbor_df['target']

    return neighbor_df['numerator'].sum() / neighbor_df['kernel'].sum() 
    


def knn_regression_cn(training_set, test_set, k, sigma, e):
    """
    create knn classifier for regression dataset with categorical and numeric values
    """
    
    test_set['predicted_target'] = np.nan
    for i in range(0, len(test_set)):
        training_set_np = np.array(training_set.loc[:, training_set.columns != 'target']).astype(float)
        test_set_subset = test_set.loc[:, (test_set.columns != 'target') & (test_set.columns != 'predicted_target') & (test_set.columns != 'e')].reset_index(drop = True)
        
        test_set_np = np.array(test_set_subset.loc[i, :]).astype(float)
        
        # use the eucledian distances
        sum_sqaured = pd.DataFrame(np.sqrt(np.sum(np.square(test_set_np - training_set_np), axis = 1)), columns = ['sum_difference'])
        label_df = training_set['target']
        
        lab_df = pd.DataFrame(label_df)
        
        neighbor_df = pd.DataFrame(pd.concat([label_df, sum_sqaured], axis = 1).nsmallest(k, 'sum_difference')).reset_index(drop = True)
        
        test_set.loc[i, 'predicted_target'] = gaussian_kernel(neighbor_df, sigma)
    test_set['e'] = e
    test_set['e'] = test_set['e'].astype(float)
    subset_df = test_set[(abs(test_set['predicted_target'] - test_set['target'])!= test_set['e'])].reset_index(drop = True)
        
    # calculate the mean squared error: mean((observed - predict) ^ 2)
    return np.mean(np.square(subset_df['predicted_target'] - subset_df['target']))



def k_classifier_cn(training_set, tuning_set, sigma_list):
    """
    calculate the classification error for each k value
    """
    
    MSE = []
    
    for k in range(1,10):
        for sigma in sigma_list:
            MSE.append([k, sigma, knn_regression_cn(training_set, tuning_set, k, sigma, 0)])
        
    final_k_df = pd.DataFrame(MSE, columns = ['K', 'Sigma', 'Mean Square Error'])
    
    return final_k_df



def cross_validation_r_c_n(k, sigma, data_subset_dict):
    """
    do five fold cross validation with the training and test set with the optimal k
    """
    for num in range(1, 6):
        test_result = []
        training_set, test_set = train_test_r(data_subset_dict, num)
        test_result.append(knn_regression_cn(training_set, test_set, k, sigma, 0))
        test_average = sum(test_result) / len(test_result)
    
    return test_average
  

data_df = abalone
tuning_set, remaining_set = train_tune_test(data_df)
data_subset_dict = create_fold_regression(remaining_set)
training_set, test_set = train_test_r(data_subset_dict, num)



knn_class = k_classifier_cn(training_set, tuning_set, [0.001, 0.01, 0.1])



knn_class


visual_k(knn_class, 'Mean Square Error', 'Abalone')


optimal_k = knn_class[knn_class['Mean Square Error'] == knn_class['Mean Square Error'].min()]['K'].values[0]
optimal_sigma = knn_class[knn_class['Mean Square Error'] == knn_class['Mean Square Error'].min()]['Sigma'].values[0]



optimal_k


optimal_sigma



knn_abalone = cross_validation_r_c_n(optimal_k, optimal_sigma, data_subset_dict)


knn_abalone

data_df = machine
tuning_set, remaining_set = train_tune_test(data_df)
data_subset_dict = create_fold_regression(remaining_set)
training_set, test_set = train_test_r(data_subset_dict, num)

knn_class = k_classifier_cn(training_set, tuning_set, [1, 10, 20])

knn_class

visual_k(knn_class, 'Mean Square Error', 'Machine')


optimal_k = knn_class[knn_class['Mean Square Error'] == knn_class['Mean Square Error'].min()]['K'].values[0]
optimal_sigma = knn_class[knn_class['Mean Square Error'] == knn_class['Mean Square Error'].min()]['Sigma'].values[0]

optimal_k

optimal_sigma

knn_machine = cross_validation_r_c_n(optimal_k, optimal_sigma, data_subset_dict)


knn_machine


def date_dfference(training_set, test_set, metric, i):
    """
    calculate the date difference 
    formula: d (d1, d2) = min(abs(d1-d2), min(d1, d2) + 12 – max(d1, d2)
    """
    train_date = np.array(training_set.loc[:, [metric]]).astype(float)
    test_date = np.array(test_set.loc[i, [metric]]).astype(float)
    train_df = pd.DataFrame(train_date, columns = ['train_' + metric])
    test_df = pd.DataFrame([test_date] * len(train_df))
    test_df.columns = ['test_' + metric]
    
    final_date_df = pd.concat([train_df, test_df], axis = 1)
    final_date_df['max_val'] = final_date_df.max(axis = 1)
    final_date_df['min_val'] = final_date_df.min(axis = 1)
    
    final_date_df['abs_diff'] = (abs(final_date_df['train_' + metric] - final_date_df['test_' + metric]))
    
    final_date_df['right'] = final_date_df['min_val'] + 12 - final_date_df['max_val'] 
    
    return final_date_df[['abs_diff', 'right']].min(axis = 1).sum()



def knn_regression_date(training_set, test_set, k, sigma, e):
    """
    knn classifier for regression dataset
    """
    test_set['predicted_target'] = np.nan
    for i in range(0, len(test_set)):
        
        training_exclude = training_set.drop(['month', 'day'], axis = 1)
        training_set_np = np.array(training_exclude.loc[:, training_exclude.columns != 'target']).astype(float)
        test_exclude = test_set.drop(['month', 'day'], axis = 1)
        test_exclude['predicted_target'] = np.nan
        
        test_drop = test_exclude.drop(['target', 'predicted_target'], axis = 1)
        
        test_set_np = np.array(test_drop.loc[:, test_drop.columns != 'e'].loc[i, :]).astype(float)   
        
        # calculate the distance beween the dates
        metric_month = 'month'
        month_sum = date_dfference(training_set, test_set, metric_month, i)
        metric_day = 'day'
        day_sum = date_dfference(training_set, test_set, metric_day, i)
        
        
        sum_sqaured = pd.DataFrame(np.sqrt(np.sum(np.square(test_set_np - training_set_np) + month_sum + day_sum, axis = 1)), columns = ['sum_difference'])
        
        label_df = training_set['target']

        lab_df = pd.DataFrame(label_df)
        
        neighbor_df = pd.DataFrame(pd.concat([label_df, sum_sqaured], axis = 1).nsmallest(k, 'sum_difference')).reset_index(drop = True)
        
        test_set.loc[i, 'predicted_target'] = gaussian_kernel(neighbor_df, sigma)
    test_set['e'] = e
    test_set['e'] = test_set['e'].astype(float)
    subset_df = test_set[(abs(test_set['predicted_target'] - test_set['target'])!= test_set['e'])].reset_index(drop = True)
        
    # calculate the mean squared error: mean((observed - predict) ^ 2)
    return np.mean(np.square(subset_df['predicted_target'] - subset_df['target']))



def k_classifier_cn_date(training_set, tuning_set, sigma_list):
    """
    calculate the classification error for each k value
    """
    
    MSE = []

    for k in range(1,10):
        for sigma in sigma_list:
            MSE.append([k, sigma, knn_regression_date(training_set, test_set, k, sigma, 0)])

            final_k_df = pd.DataFrame(MSE, columns = ['K', 'Sigma', 'Mean Square Error'])
    
    return final_k_df



def cross_validation_date(k, sigma, data_subset_dict):
    """
    do five fold cross validation with the training and test set with the optimal k
    """
    for num in range(1, 6):
        test_result = []
        training_set, test_set = train_test_r(data_subset_dict, num)
        test_result.append(knn_regression_date(training_set, test_set, k, sigma, 0))
        test_average = sum(test_result) / len(test_result)
    
    return test_average



data_df = forestfires
tuning_set, remaining_set = train_tune_test(data_df)
data_subset_dict = create_fold_regression(remaining_set)
training_set, test_set = train_test_r(data_subset_dict, num)


knn_class = k_classifier_cn_date(training_set, tuning_set, [1, 10, 20])



knn_class




visual_k(knn_class, 'Mean Square Error', 'Forestfire')


optimal_k = knn_class[knn_class['Mean Square Error'] == knn_class['Mean Square Error'].min()]['K'].values[0]
optimal_sigma = knn_class[knn_class['Mean Square Error'] == knn_class['Mean Square Error'].min()]['Sigma'].values[0]


optimal_k



optimal_sigma



knn_forestfire = cross_validation_date(optimal_k, optimal_sigma, data_subset_dict)


knn_forestfire


# #### Edited Nearest Neighbor
# 
# We use incremental editing process to remove observations from a training set that are misclassified using the rest of the instances in the training set
# 
# * Split the dataset into training set, validation set and test set
# * Use k = 1 for editing`
# 1. Select one instance from the training set, classify the example using the rest of the training set. If the classification prediction is incorrect, remove the instance from the training set, apply the knn algorithm with the reduced training set on the validation set and get the classification error
# 2. Repeat the step above until the classification error on the validation set starts to increase
# 3. Save the k value, the classification error and the reduced training set



def enn_edit(training_set, tuning_set, class_function, class_var, k = 1):
    
    k_dict = {}

    # shuffle the dataset
    training_set = shuffle(training_set).reset_index(drop = True)

    # independent validation set
    tuning_set = shuffle(tuning_set).reset_index(drop = True)

    index_list = []
    
    row_num = len(training_set)
    training_copy = training_set
    
    for i in range(0, len(training_set)):
        
        # select an instance from the training set
        instance = np.array(training_set.loc[i, training_set.columns != class_var]).astype(float)
        # select the remaining data in the training set
        t_test = np.array(training_set.loc[training_set.index != i, training_set.columns != class_var]).astype(float)
        
        # calculate the sum of squared errors
        sum_sqaured = pd.DataFrame(np.sqrt(np.sum(np.square(instance - t_test), axis = 1)), columns = ['sum_difference'])

        sum_sqaured.index = list(training_set[training_set.index != i].index)

        label_df = training_set.loc[training_set.index != i, class_var]

        # get the neighbors with the top k neighbors with lowest sum of square errors
        neighbor_df = pd.DataFrame(pd.concat([label_df, sum_sqaured], axis = 1).nsmallest(k, 'sum_difference').groupby([class_var])[class_var].count()).rename(columns = {class_var: '# observations'}).reset_index()

        # get the prediction
        predicted_class = neighbor_df.sort_values(by = '# observations', ascending = False).reset_index(drop = True).loc[0, class_var]
   
        
        v_error = class_function(training_set.reset_index(drop = True), tuning_set, k)
        
    
        if ((i == (row_num - 1)) and (training_set.loc[i, class_var] == predicted_class)):
            k_dict[k] = [training_copy[~training_copy.index.isin([index_list])]]
        
        elif training_set.loc[i, class_var] != predicted_class :
            # remove misclassified instances

            training_set = training_set.drop(i)
            index_list.append(i)
            
            # calculate the new classification error after removing the misclassified instances
            new_error = knn_classification_numeric(training_set.reset_index(drop = True), tuning_set, k)

            # if error rate increases, stop removing the instances
            if new_error > v_error or i == (row_num - 1):
                before_drop = training_set
                index_list.remove(i)

                k_dict[k] = [before_drop[~before_drop.index.isin([index_list])]]
                break
            else:
                v_error = new_error
    
    if k_dict == {}:
        new_training_set = pd.DataFrame([])
    else:
        output = k_dict[k]
        new_training_set = pd.DataFrame(output[0]).reset_index(drop = True)
    return new_training_set
    





def enn(training_set, tuning_set, edit_func, function, class_var):
    """
    apply the enn algorithm to the tuning set to find the optimal k
    """

    k_accuracy = []
    reduced_set_shape = {}
    for k_val in range(1,10):
        new_training_set = edit_func(training_set, tuning_set, knn_classification_numeric, class_var, k = 1)
        reduced_set_shape[k_val] = new_training_set
        tuning_set = shuffle(tuning_set).reset_index(drop = True)
        test_result = function(new_training_set, tuning_set, k_val)
        k_accuracy.append([k_val, test_result])
    final_k_df = pd.DataFrame(k_accuracy, columns = ['K', 'Classification Error'])

    return final_k_df, reduced_set_shape





def visual_k_enn(enn_class, metric, dname):
    """
    visualize the elbow plot
    """
    return sns.lineplot(x = enn_class['K'], y = enn_class[metric]).set_title('e-NN Algorithm Classification Error For Various K - ' + dname)





def cross_validation_enn(k, reduced_set, data_subset_dict, function):
    """
    do five fold cross validation with the training and test set with the optimal k
    """
    
    reduced_training_set = reduced_set[k]
    
    test_result = []
    for num in range(1, 6):
        training_set, test_set = train_test_c(data_subset_dict, num)
        test_result.append(function(reduced_training_set, test_set, k))
    test_average = sum(test_result) / len(test_result)
    
    return test_average





# implement enn algorithm on glass dataset
data_df = glass
num = 1
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)
training_set, test_set = train_test_c(data_subset_dict, num)
enn_class, reduced_set = enn(training_set, tuning_set, enn_edit, knn_classification_numeric, 'class')

enn_class


visual_k_enn(enn_class, 'Classification Error', 'Glass')


optimal_k = enn_class[enn_class['Classification Error'] == enn_class['Classification Error'].min()]['K'].values[0]


optimal_k


enn_glass = cross_validation_enn(optimal_k, reduced_set, data_subset_dict, knn_classification_numeric)


enn_glass


# implement enn algorithm on segmentation dataset
data_df = seg_final_df
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)
training_set, test_set = train_test_c(data_subset_dict, num)
enn_class, reduced_set = enn(training_set, tuning_set, enn_edit, knn_classification_numeric, 'class')


enn_class

visual_k_enn(enn_class, 'Classification Error', 'Segmentation')

optimal_k = enn_class[enn_class['Classification Error'] == enn_class['Classification Error'].min()]['K'].values[0]


optimal_k

enn_segmentation = cross_validation_enn(optimal_k, reduced_set, data_subset_dict, knn_classification_numeric)

enn_segmentation


def enn_edit_cat(training_set, tuning_set, k = 1):
    
    k_dict = {}

    # shuffle the dataset
    training_set = shuffle(training_set).reset_index(drop = True)

    # independent validation set
    tuning_set = shuffle(tuning_set).reset_index(drop = True)

    index_list = []
    
    #print('training_set size', len(training_set))
    
    row_num = len(training_set)
    training_copy = training_set
    
    for i in range(0, len(training_set)):
        instance = training_set.loc[i, training_set.columns != 'class']
        total_distance = {}
        remaining = training_set.loc[training_set.index != i, training_set.columns != 'class'].reset_index(drop = True)
        for j in range(len(remaining)):
            distance = []
            for col in remaining.columns.unique():
                train_val = remaining.loc[j, col]
                test_val = instance[col]
                distance.append(dict_df[col][train_val][test_val])
                total_distance[j] = np.sqrt(np.sum(distance))
        sum_squared = pd.DataFrame.from_dict([total_distance]).T
        sum_squared.columns = ['sum_difference']
        sum_squared.index = list(training_set[training_set.index != i].index)

        label_df = training_set.loc[training_set.index != i, 'class']

        # get the neighbors with the top k neighbors with lowest sum of square errors
        neighbor_df = pd.DataFrame(pd.concat([label_df, sum_squared], axis = 1).nsmallest(k, 'sum_difference').groupby(['class'])['class'].count()).rename(columns = {'class': '# observations'}).reset_index()

        # get the prediction
        predicted_class = neighbor_df.sort_values(by = '# observations', ascending = False).reset_index(drop = True).loc[0, 'class']
        #print('predicted class is', predicted_class)
        
        v_error = knn_classification_categorical(training_set.reset_index(drop = True), tuning_set, k)
        
        #print('real class is', training_set.loc[i, 'class'])
        
        if ((i == (row_num - 1)) and (training_set.loc[i, 'class'] == predicted_class)):
            k_dict[k] = [training_copy[~training_copy.index.isin([index_list])]]
        
        elif training_set.loc[i, 'class'] != predicted_class :
            #print('misclassification!!!')
            # remove misclassified instances
            #print('remove this row', i)
            training_set = training_set.drop(i)
            index_list.append(i)
            
            # calculate the new classification error after removing the misclassified instances
            new_error = knn_classification_categorical(training_set.reset_index(drop = True), tuning_set, k)

            # if error rate increases, stop removing the instances
            if new_error > v_error or i == (row_num - 1):
                before_drop = training_set
                index_list.remove(i)
                #print('stop removing')
                #print('index_list', index_list)
                k_dict[k] = [before_drop[~before_drop.index.isin([index_list])]]
                break
            else:
                v_error = new_error
    #print(k_dict)
    
    if k_dict == {}:
        new_training_set = pd.DataFrame([])
    else:
        output = k_dict[k]
        new_training_set = pd.DataFrame(output[0]).reset_index(drop = True)
    return new_training_set


# implement enn algorithm on vote dataset
data_df = vote
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)

training_set, test_set = train_test_c(data_subset_dict, num)

dict_df = {}

for c in data_df.loc[:, data_df.columns != 'class'].columns.unique():
    dict_df[c] = distance_df(c, data_df)


enn_class, reduced_set = enn(training_set, tuning_set, enn_edit_cat, knn_classification_categorical)


enn_class


visual_k_enn(enn_class, 'Classification Error', 'Vote')


optimal_k = enn_class[enn_class['Classification Error'] == enn_class['Classification Error'].min()]['K'].values[0]


optimal_k

enn_vote = cross_validation_enn(k, reduced_set, data_subset_dict, knn_classification_categorical)


enn_vote


def enn_edit_r(training_set, tuning_set, class_function, class_var, sigma, e, k = 1):
    
    k_dict = {}

    # shuffle the dataset
    training_set = shuffle(training_set).reset_index(drop = True)

    # independent validation set
    tuning_set = shuffle(tuning_set).reset_index(drop = True)

    index_list = []
    
    #print('training_set size', len(training_set))
    
    row_num = len(training_set)
    training_copy = training_set
    
    for i in range(0, len(training_set)):
        
        #print('row num', i)
        # select an instance from the training set
        instance = np.array(training_set.loc[i, training_set.columns != class_var]).astype(float)
        # select the remaining data in the training set
        t_test = np.array(training_set.loc[training_set.index != i, training_set.columns != class_var]).astype(float)
        
        # calculate the sum of squared errors
        sum_sqaured = pd.DataFrame(np.sqrt(np.sum(np.square(instance - t_test), axis = 1)), columns = ['sum_difference'])

        sum_sqaured.index = list(training_set[training_set.index != i].index)

        label_df = training_set.loc[training_set.index != i, class_var]

        # get the neighbors with the top k neighbors with lowest sum of square errors
        neighbor_df = pd.DataFrame(pd.concat([label_df, sum_sqaured], axis = 1).nsmallest(k, 'sum_difference').groupby([class_var])[class_var].count()).rename(columns = {class_var: '# observations'}).reset_index()

        # get the prediction
        predicted_class = neighbor_df.sort_values(by = '# observations', ascending = False).reset_index(drop = True).loc[0, class_var]
        #print('predicted class is', predicted_class)
        
        v_error = class_function(training_set.reset_index(drop = True), tuning_set, k, sigma, e)
        
        
        if ((i == (row_num - 1)) and (training_set.loc[i, class_var] == predicted_class)):
            k_dict[k] = [training_copy[~training_copy.index.isin([index_list])]]
        
        elif training_set.loc[i, class_var] != predicted_class :
  
            training_set = training_set.drop(i)
            index_list.append(i)
            
            # calculate the new classification error after removing the misclassified instances
            new_error = class_function(training_set.reset_index(drop = True), tuning_set, k, sigma, e)

            # if error rate increases, stop removing the instances
            if new_error > v_error or i == (row_num - 1):
                before_drop = training_set
                index_list.remove(i)

                k_dict[k] = [before_drop[~before_drop.index.isin([index_list])]]
                break
            else:
                v_error = new_error
    #print(k_dict)
    
    if k_dict == {}:
        new_training_set = pd.DataFrame([])
    else:
        output = k_dict[k]
        new_training_set = pd.DataFrame(output[0]).reset_index(drop = True)
    return new_training_set





def enn_reg(training_set, tuning_set, edit_func, function, sigma, e_list, class_var):
    """
    apply the enn algorithm to the tuning set to find the optimal k
    """

    MSE = []
    reduced_set_shape = {}
    for k_val in range(1,10):
        for e in e_list:
            new_training_set = edit_func(training_set, tuning_set, knn_regression_cn, class_var, sigma, e, k = 1)
            reduced_set_shape[k_val] = new_training_set
            tuning_set = shuffle(tuning_set).reset_index(drop = True)
            test_result = function(training_set, test_set, k_val, sigma, e)
            MSE.append([k_val, e, test_result])
    final_k_df = pd.DataFrame(MSE, columns = ['K', 'Epsilon', 'Mean Square Error'])

    return final_k_df, reduced_set_shape





def cv_enn_reg(k, e, reduced_set, data_subset_dict, function, sigma):
    """
    do five fold cross validation with the training and test set with the optimal k
    """
    
    reduced_training_set = reduced_set[k]
    
    test_result = []
    for num in range(1, 6):
        training_set, test_set = train_test_r(data_subset_dict, num)
        test_result.append(function(training_set, test_set, k, sigma, e))
    test_average = sum(test_result) / len(test_result)
    
    return test_average





# implement enn algorithm on abalone dataset

data_df = abalone
num = 1
tuning_set, remaining_set = train_tune_test(data_df)
data_subset_dict = create_fold_regression(remaining_set)
training_set, test_set = train_test_r(data_subset_dict, num)

sigma = 0.01
e_list = [1, 3, 5]
enn_class, reduced_set = enn_reg(training_set, tuning_set, enn_edit_r, knn_regression_cn, sigma, e_list, 'target')


visual_k_enn(enn_class, 'Mean Square Error', 'Abalone')


optimal_k = enn_class[enn_class['Mean Square Error'] == enn_class['Mean Square Error'].min()]['K'].values[0]
optimal_e = enn_class[enn_class['Mean Square Error'] == enn_class['Mean Square Error'].min()]['Epsilon'].values[0]

optimal_k


optimal_e

enn_glass = cv_enn_reg(optimal_k, optimal_e, reduced_set, data_subset_dict, knn_regression_cn, sigma)


enn_glass


# implement enn algorithm on machine dataset

data_df = machine
tuning_set, remaining_set = train_tune_test(data_df)
data_subset_dict = create_fold_regression(remaining_set)
training_set, test_set = train_test_r(data_subset_dict, num)


sigma = 10
e_list = [5, 10, 20]
enn_class, reduced_set = enn_reg(training_set, tuning_set, enn_edit_r, knn_regression_cn, sigma, e_list, 'target')

enn_class


visual_k_enn(enn_class, 'Mean Square Error', 'Machine')

optimal_k = enn_class[enn_class['Mean Square Error'] == enn_class['Mean Square Error'].min()]['K'].values[0]
optimal_e = enn_class[enn_class['Mean Square Error'] == enn_class['Mean Square Error'].min()]['Epsilon'].values[0]


optimal_k


optimal_e


enn_machine = cv_enn_reg(optimal_k, optimal_e, reduced_set, data_subset_dict, knn_regression_cn, sigma)


enn_machine


# implement enn algorithm on forestfires dataset
data_df = forestfires
tuning_set, remaining_set = train_tune_test(data_df)
data_subset_dict = create_fold_regression(remaining_set)
training_set, test_set = train_test_r(data_subset_dict, num)

sigma = 1
e_list = [1, 3, 5]
enn_class, reduced_set = enn_reg(training_set, tuning_set, enn_edit_r, knn_regression_date, sigma, e_list, 'target')


enn_class


visual_k_enn(enn_class, 'Mean Square Error', 'Forestfires')


optimal_k = enn_class[enn_class['Mean Square Error'] == enn_class['Mean Square Error'].min()]['K'].values[0]
optimal_e = enn_class[enn_class['Mean Square Error'] == enn_class['Mean Square Error'].min()]['Epsilon'].values[0]


optimal_k


optimal_e

enn_forestfires = cv_enn_reg(optimal_k, optimal_e, reduced_set, data_subset_dict, knn_regression_cn, sigma)

enn_forestfires


# #### Condensed Nearest Neighbor
# 1. Start with an empty set Z
# 2. The initial point added to Z is considered as misclassified
# 3. For each instance in the training set, find the nearest point in the set Z, if the classes are matched, we go to the next instance. If their classes are mismatched, add the instance to Z and remove it from the training set
# 4. Input Z as a training set into the knn classification algorithm

def cnn_df(training_set, class_var):
    """
    use cnn algorithm to build set Z
    """
    
    Z = {}

    # create an initial element in Z
    Z[0] = [training_set.loc[0, class_var], training_set.iloc[0, training_set.columns != class_var]]

    i = 1
    original_length = len(training_set)

    # iterate through all the elements in the training set
    while i < (original_length):
        # get the instance value and class
        instance = np.array(training_set.loc[i, training_set.columns != class_var]).astype(float)
        instance_class = training_set.loc[i, training_set.columns == class_var][0]

        c_error_list = {}

        # calculate the eucledian distance between the instance and each element in Z
        for key, items in Z.items():
            c_error_list[key] = np.sqrt(np.sum(np.square(np.array(items[1]).astype(float) - instance)))

        # if classes are matched
        if Z[min(c_error_list, key = c_error_list.get)][0] == instance_class:
            i = i + 1
        else:
            # if classes are not matched, drop the instance from the training set
            Z[i] = [training_set.loc[i, class_var], training_set.loc[i, training_set.columns != class_var]]
            training_set = training_set.drop(i)
            i = i + 1

    Z_df = pd.DataFrame()

    # store the instances in Z in the dataframe
    for key in Z:
        d_df = pd.DataFrame(Z[key][1]).T.reset_index(drop = True)
        c_df = pd.DataFrame([Z[key][0]]).T
        f_df = pd.concat([d_df, c_df], axis = 1).rename(columns = {0: class_var})
        Z_df = pd.concat([Z_df, f_df], axis = 0)

    Z_df = Z_df.reset_index(drop = True)
    
    return Z_df


def cnn(training_set, tuning_set, function, class_var):
    """
    create cnn classifier 
    """

    k_accuracy = []
    reduced_set_shape = {}
    for k_val in range(1,10):
        new_training_set = cnn_df(training_set, class_var)
        reduced_set_shape[k_val] = new_training_set
        tuning_set = shuffle(tuning_set).reset_index(drop = True)
        test_result = function(new_training_set, tuning_set, k_val)
        k_accuracy.append([k_val, test_result])
    final_k_df = pd.DataFrame(k_accuracy, columns = ['K', 'Classification Error'])

    return final_k_df, reduced_set_shape


def cross_validation_cnn(k, reduced_set, data_subset_dict, function):
    """
    do five fold cross validation with the training and test set with the optimal k
    """
    
    reduced_training_set = reduced_set[k]
    
    test_result = []
    for num in range(1, 6):
        training_set, test_set = train_test_c(data_subset_dict, num)
        test_result.append(function(reduced_training_set, test_set, k))
    test_average = sum(test_result) / len(test_result)
    
    return test_average



def visual_k_cnn(cnn_class, metric, dname):
    """
    visualize the elbow plot
    """
    return sns.lineplot(x = cnn_class['K'], y = cnn_class[metric]).set_title('c-NN Algorithm Classification Error For Various K - ' + dname)


# implement cnn algorithm on glass dataset
data_df = glass
num = 1
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)
training_set, test_set = train_test_c(data_subset_dict, num)
cnn_class, reduced_set = cnn(training_set, tuning_set, knn_classification_numeric, 'class')

cnn_class

visual_k_cnn(cnn_class, 'Classification Error', 'Glass')


optimal_k = cnn_class[cnn_class['Classification Error'] == cnn_class['Classification Error'].min()]['K'].values[0]

cnn_glass = cross_validation_cnn(optimal_k, reduced_set, data_subset_dict, knn_classification_numeric)


cnn_glass

# implement cnn algorithm on segmentation dataset
data_df = seg_final_df
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)
training_set, test_set = train_test_c(data_subset_dict, num)
cnn_class, reduced_set = cnn(training_set, tuning_set, knn_classification_numeric, 'class')

cnn_class

visual_k_cnn(cnn_class, 'Classification Error', 'Segmentation')


optimal_k = cnn_class[cnn_class['Classification Error'] == cnn_class['Classification Error'].min()]['K'].values[0]

optimal_k

cnn_segmentation = cross_validation_cnn(optimal_k, reduced_set, data_subset_dict, knn_classification_numeric)

cnn_segmentation

def cnn_df_cat(training_set, class_var):
    
    Z = {}

    # create an initial element in Z
    Z[0] = [training_set.loc[0, 'class'], training_set.loc[0, training_set.columns != 'class']]

    i = 1
    original_length = len(training_set)

    # iterate through all the elements in the training set
    while i < (original_length):
        # get the instance value and class
        instance = training_set.loc[i, training_set.columns != 'class']
        instance_class = training_set.loc[i, training_set.columns == 'class'][0]

        c_error_list = {}

        # calculate the eucledian distance between the instance and each element in Z
        for key, items in Z.items():
            distance = []
            for i in range(len(instance)):
                test_val = instance[i]
                train_val = items[1][i]
                distance.append(dict_df[instance.index[i]][train_val][test_val])
            c_error_list[key] = np.sqrt(np.sum(distance))

        # if classes are matched
        if Z[min(c_error_list, key = c_error_list.get)][0] == instance_class:
            i = i + 1
        else:
            # if classes are not matched, drop the instance from the training set
            Z[i] = [training_set.loc[i, 'class'], training_set.loc[i, training_set.columns != 'class']]
            training_set = training_set.drop(i)
            i = i + 1

    Z_df = pd.DataFrame()

    # store the instances in Z in the dataframe
    for key in Z:
        d_df = pd.DataFrame(Z[key][1]).T.reset_index(drop = True)
        c_df = pd.DataFrame([Z[key][0]]).T
        f_df = pd.concat([d_df, c_df], axis = 1).rename(columns = {0: 'class'})
        Z_df = pd.concat([Z_df, f_df], axis = 0)

    Z_df = Z_df.reset_index(drop = True)
    
    return Z_df


def cnn_cat(training_set, tuning_set):

    k_accuracy = []
    reduced_set_shape = {}
    for k_val in range(1,10):
        new_training_set = cnn_df_cat(training_set)
        reduced_set_shape[k_val] = new_training_set
        tuning_set = shuffle(tuning_set).reset_index(drop = True)
        test_result = knn_classification_categorical(new_training_set, tuning_set, k_val)
        k_accuracy.append([k_val, test_result])
    final_k_df = pd.DataFrame(k_accuracy, columns = ['K', 'Classification Error'])

    return final_k_df, reduced_set_shape


# implement cnn algorithm on vote dataset

data_df = vote
num = 1
tuning_set, remaining_set = tune_remaining(data_df)
data_subset_dict = create_fold_classification(remaining_set)
training_set, test_set = train_test_c(data_subset_dict, num)

dict_df = {}

for c in data_df.loc[:, data_df.columns != 'class'].columns.unique():
    dict_df[c] = distance_df(c, data_df)

cnn_df_cat(training_set)


cnn_class, reduced_set = cnn_cat(training_set, tuning_set)

cnn_class

visual_k_cnn(cnn_class, 'Classification Error', 'Vote')

optimal_k = cnn_class[cnn_class['Classification Error'] == cnn_class['Classification Error'].min()]['K'].values[0]


optimal_k


cnn_vote = cross_validation_cnn(k, reduced_set, data_subset_dict, knn_classification_categorical)

cnn_vote

def cnn_reg(training_set, tuning_set, edit_func, function, sigma, e_list, class_var):
    """
    apply the cnn algorithm to the tuning set to find the optimal k
    """

    MSE = []
    reduced_set_shape = {}
    for k_val in range(1,10):
        for e in e_list:
            new_training_set = edit_func(training_set, class_var)
            reduced_set_shape[k_val] = new_training_set
            tuning_set = shuffle(tuning_set).reset_index(drop = True)
            test_result = function(training_set, test_set, k_val, sigma, e)
            MSE.append([k_val, e, test_result])
    final_k_df = pd.DataFrame(MSE, columns = ['K', 'Epsilon', 'Mean Square Error'])

    return final_k_df, reduced_set_shape

def cv_cnn_reg(k, e, reduced_set, data_subset_dict, function, sigma):
    """
    do five fold cross validation with the training and test set with the optimal k
    """
    
    reduced_training_set = reduced_set[k]
    
    test_result = []
    for num in range(1, 6):
        training_set, test_set = train_test_r(data_subset_dict, num)
        test_result.append(function(training_set, test_set, k, sigma, e))
    test_average = sum(test_result) / len(test_result)
    
    return test_average


# implement cnn algorithm on abalone dataset

data_df = abalone
num = 1
tuning_set, remaining_set = train_tune_test(data_df)
data_subset_dict = create_fold_regression(remaining_set)
training_set, test_set = train_test_r(data_subset_dict, num)

sigma = 0.01
e_list = [1, 3, 5]
cnn_class, reduced_set = cnn_reg(training_set, tuning_set, cnn_df, knn_regression_cn, sigma, e_list, 'target')


visual_k_cnn(cnn_class, 'Mean Square Error', 'Abalone')


optimal_k = cnn_class[cnn_class['Mean Square Error'] == cnn_class['Mean Square Error'].min()]['K'].values[0]
optimal_e = cnn_class[cnn_class['Mean Square Error'] == cnn_class['Mean Square Error'].min()]['Epsilon'].values[0]

optimal_k


optimal_e


cnn_glass = cv_cnn_reg(optimal_k, optimal_e, reduced_set, data_subset_dict, knn_regression_cn, sigma)


cnn_glass


# implement cnn algorithm on machine dataset

data_df = machine
num = 1
tuning_set, remaining_set = train_tune_test(data_df)
data_subset_dict = create_fold_regression(remaining_set)
training_set, test_set = train_test_r(data_subset_dict, num)

sigma = 10
e_list = [5, 10, 20]
cnn_class, reduced_set = cnn_reg(training_set, tuning_set, cnn_df, knn_regression_cn, sigma, e_list, 'target')

cnn_class

visual_k_cnn(cnn_class, 'Mean Square Error', 'Machine')

optimal_k = cnn_class[cnn_class['Mean Square Error'] == cnn_class['Mean Square Error'].min()]['K'].values[0]
optimal_e = cnn_class[cnn_class['Mean Square Error'] == cnn_class['Mean Square Error'].min()]['Epsilon'].values[0]


optimal_k



optimal_e


cnn_machine = cv_cnn_reg(optimal_k, optimal_e, reduced_set, data_subset_dict, knn_regression_cn, sigma)

cnn_machine


# implement cnn algorithm on forestfires dataset

data_df = forestfires
tuning_set, remaining_set = train_tune_test(data_df)
data_subset_dict = create_fold_regression(remaining_set)
training_set, test_set = train_test_r(data_subset_dict, num)

sigma = 1
e_list = [1, 3, 5]
cnn_class, reduced_set = cnn_reg(training_set, tuning_set, cnn_df, knn_regression_date, sigma, e_list, 'target')

cnn_class

visual_k_cnn(cnn_class, 'Mean Square Error', 'Forestfires')

optimal_k = cnn_class[cnn_class['Mean Square Error'] == cnn_class['Mean Square Error'].min()]['K'].values[0]
optimal_e = cnn_class[cnn_class['Mean Square Error'] == cnn_class['Mean Square Error'].min()]['Epsilon'].values[0]

optimal_k

optimal_e

cnn_forestfires = cv_cnn_reg(optimal_k, optimal_e, reduced_set, data_subset_dict, knn_regression_cn, sigma)
cnn_forestfires













