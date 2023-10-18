#!/usr/bin/env python
# coding: utf-8

# # lead score Conversion

# In[1]:


# Supress unnecessary warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Import the NumPy and Pandas packages

import numpy as np
import pandas as pd


# In[3]:


#reading the data set
leads = pd.read_csv("C:/Users/Soumya/Desktop/Data Science/project/Leads.csv")


# In[4]:


leads.head()


# In[5]:


leads.shape


# In[6]:


leads.columns


# In[7]:


leads.describe


# In[8]:


leads.describe()


# In[9]:


leads.info()


# ## Step 1: Data Cleaning and Preparation

# In[10]:


# Check the number of missing values in each column

leads.isnull().sum()


# In[11]:


# Droping  all the columns in which greater than 3000 missing values are present

for col in leads.columns:
    if leads[col].isnull().sum() > 3000:
        leads.drop(col, 1, inplace=True)


# In[12]:


leads.isnull().sum()


# In[13]:


leads.drop(['City'], axis = 1, inplace = True)


# In[14]:


leads.drop(['Country'], axis = 1, inplace = True)


# In[15]:


round(100*(leads.isnull().sum()/len(leads.index)), 2)


# In[16]:


leads.isnull().sum()


# In[17]:


#Get the value counts of all the columns

for column in leads:
   print(leads[column].astype('category').value_counts())
   print('___________________________________________________')


# In[18]:


leads['Lead Profile'].astype('category').value_counts()


# In[19]:


leads['How did you hear about X Education'].value_counts()


# In[20]:


leads['Specialization'].value_counts()


# In[21]:


leads['Lead Source'].value_counts()


# In[22]:


leads.drop(['Lead Profile', 'How did you hear about X Education'], axis = 1, inplace = True)


# In[23]:


leads.drop(['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
            'Update me on Supply Chain Content', 'Get updates on DM Content', 
            'I agree to pay the amount through cheque'], axis = 1, inplace = True)


# In[24]:


leads.isnull().sum()


# In[25]:


leads['What matters most to you in choosing a course'].value_counts()


# In[26]:


leads.drop(['What matters most to you in choosing a course'], axis = 1, inplace=True)


# In[27]:


leads.isnull().sum()


# In[28]:


leads = leads[~pd.isnull(leads['What is your current occupation'])]


# In[29]:


leads.isnull().sum()


# In[30]:


leads = leads[~pd.isnull(leads['TotalVisits'])]


# In[31]:


leads.isnull().sum()


# In[32]:


leads = leads[~pd.isnull(leads['Lead Source'])]


# In[33]:


leads.isnull().sum()


# In[34]:


leads = leads[~pd.isnull(leads['Specialization'])]


# In[35]:


leads.isnull().sum()


# In[36]:


print(len(leads.index))
print(len(leads.index)/9240)


# In[37]:


leads.shape


# In[38]:


leads.head


# In[39]:


leads.head()


# In[40]:


leads.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[41]:


leads.head()


# In[42]:


#creating dummy variables
temp = leads.loc[:, leads.dtypes == 'object']
temp.columns


# In[43]:


# Creating dummy variables using the 'get_dummies' command
dummy = pd.get_dummies(leads[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                              'What is your current occupation','A free copy of Mastering The Interview', 
                              'Last Notable Activity']], drop_first=True)

# Add the results to the master dataframe
leads = pd.concat([leads, dummy], axis=1)


# In[44]:


# Creating dummy variable separately for the variable 'Specialization' since it has the level 'Select' which is useless so we
# drop that level by specifying it explicitly

dummy_spl = pd.get_dummies(leads['Specialization'], prefix = 'Specialization')
dummy_spl = dummy_spl.drop(['Specialization_Select'], 1)
leads = pd.concat([leads, dummy_spl], axis = 1)


# In[45]:


# Drop the variables for which the dummy variables have been created

leads = leads.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                   'Specialization', 'What is your current occupation',
                   'A free copy of Mastering The Interview', 'Last Notable Activity'], 1)


# In[46]:



leads.head()


# In[47]:


leads.shape


# ### Test-Train Split
# 
# The next step is to split the dataset into training an testing sets.

# In[48]:


# Import the required library

from sklearn.model_selection import train_test_split


# In[49]:


X = leads.drop(['Converted'], 1)
X.head()


# In[50]:


# Put the target variable in y

y = leads['Converted']

y.head()


# In[51]:


# Split the dataset into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# # Split the dataset into 70% train and 30% test
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# In[52]:


# Import MinMax scaler

from sklearn.preprocessing import MinMaxScaler


# In[53]:


# Scale the three numeric features present in the dataset

scaler = MinMaxScaler()

X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

X_train.head()


# ### Looking at the correlations
# 
# Let's now look at the correlations. Since the number of variables are pretty high, it's better that we look at the table instead of plotting a heatmap

# In[54]:


leads.corr()


# # Step 2: Model Building

# In[55]:


# Importing 'LogisticRegression' and create a LogisticRegression object
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()  # Instantiate a logistic regression model

rfe = RFE(estimator=logreg, n_features_to_select=15)
rfe.fit(X_train, y_train)


# In[56]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[57]:


col = X_train.columns[rfe.support_]


# In[58]:


col


# In[59]:


X_train = X_train[col]


# In[60]:


import statsmodels.api as sm


# In[61]:


X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[62]:


# Importing 'variance_inflation_factor'

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[63]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[64]:


X_train.drop('Lead Source_Reference', axis = 1, inplace = True)


# In[65]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[66]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[67]:


X_train.drop('Last Notable Activity_Had a Phone Conversation', axis = 1, inplace = True)


# In[68]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[69]:


X_train.drop('What is your current occupation_Housewife', axis = 1, inplace = True)


# In[70]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[71]:


X_train.drop('What is your current occupation_Working Professional', axis = 1, inplace = True)


# In[72]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[73]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Step 3: Model Evaluation

# In[74]:


# Using 'predict' to predict the probabilities on the train set

y_train_pred = res.predict(sm.add_constant(X_train))
y_train_pred[:10]


# In[75]:


#Reshaping it into an array

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[76]:


# Create a new dataframe containing the actual conversion flag and the probabilities predicted by the model

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final


# In[77]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[78]:


# Import metrics from sklearn for evaluation

from sklearn.metrics import confusion_matrix,plot_confusion_matrix,accuracy_score,roc_curve,roc_auc_score


# In[79]:


confusion = confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[80]:


# Let's check the overall accuracy

print(accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[81]:


# Let's evaluate the other metrics as well

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[82]:


# Calculate the sensitivity

TP/(TP+FN)


# In[83]:


# Calculate the specificity

TN/(TN+FP)


#  ### Finding the Optimal Cutoff

# In[84]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[85]:


fpr, tpr, thresholds = roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[86]:


import matplotlib.pyplot as plt


# In[87]:


# Calling  the ROC function

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[88]:


#  creating columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[89]:


# creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[90]:


#Let's plot it as well

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[91]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.42 else 0)

y_train_pred_final.head()


# In[92]:


# Let's check the accuracy now
accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[93]:


# Let's create the confusion matrix once again

confusion2 = confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[94]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[95]:


# Calculate Sensitivity

TP/(TP+FN)


# In[96]:


# Calculate Specificity

TN/(TN+FP)


# In[ ]:




