#!/usr/bin/env python
# coding: utf-8

# # **Hypertension patients classification with the use of ML methods**
# 
#     
# ##### • *Author:* Hubert Kycia
#     
# ##### • *Date:* 30.05.2023
# 

# <img src="https://www.thesun.co.uk/wp-content/uploads/2021/11/NINTCHDBPICT000728842198.jpg?w=1920" alt="Obrazek" width="800" >

# <a id="0"></a> <br>
#  ## Table of Contents  
# 1. [Introduction](#1) 
# 1. [Exploratory Data Analysis](#2)     
#     2.1. [Dataset description](#3)    
#     2.2. [EDA: relationships between variables](#4)
# 1. [ML algorithms for classification](#5) <br>
#     3.1. [Decision Tree](#6)    
#     3.2. [Random Forest](#7)<br>
#     3.3. [SVM](#8)    
#     3.4. [Bagging](#9)<br>
#     3.5. [Gradient Boosting](#10)    
#     3.6. [XGBoost](#11)
# 1. [Conclusion](#12)     

# ## 1. Introduction <a class="anchor" id="1"></a>
# <div style="text-align:justify;line-height: 1.6; margin-top:10px;">
# Hypertension is a sustained elevation of blood pressure, which is 140/90 mmHg or higher. Hypertension usually does not cause symptoms for many years and, if the blood pressure value is not regularly controlled, it is detected when complications affecting various organs (e.g. heart, kidneys, brain) occur. Treatment consists of lifestyle modification - adequate physical activity and maintaining a normal body weight, as well as taking blood pressure-lowering medication. Among most patients, no specific cause for the development of hypertension is identified. Many factors may contribute to its elevated values:
# </div>
#     
# - hereditary susceptibility,
# 
# - obesity,
# 
# - unhealthy diet (high salt intake),
# 
# - the ageing process,
# 
# - mental stress,
# 
# - sedentary lifestyle.
#     
# <div style="width: 100%; text-align:justify;line-height: 1.6; margin-top:10px;">  
# Hypertension is a common disease. In the chart below, we can observe the percentage of people who suffer from hypertension in different European countries in 2019. According to the data gathered by Eurostat, the probability that a random person in Europe has hypertension is higher than 0.2. In most considered countries, the percentage of people suffering from hypertension is higher among women. The highest percentage of ill people was observed in Croatia (37%), Latwia (32%), Hungary (32%).
# 
# </div>

# <img src="https://ec.europa.eu/eurostat/documents/4187653/11581523/high+blood+pressure.png" alt="Hypertension" width="800" text-aling=center>

# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# It can be seen that the number of people with hypertension is concerningly high, accounting for more than 30% of the adult population (15+) in many European countries. This means that the problem associated with the prevalence of arterial hypertension is a serious one, affecting nearly one third of adults. What is more, the percentage of sick people does not show a downward trend. The average prevalence of hypertension in the global population is expected to rise to 29.2% in 2025 - at that point the problem will affect around 1.65 billion people.
# </div>
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The aim of the project is to diagnose the factors that have the greatest impact on the risk of having hypertension and use machine learning methods for classification which assign probability to each patient of having hypertension based on patient data. In this study, methods such as Decision Tree, SVM, Random Forest, Bagging, Gradient Boosting and XGBoost were used. In addition, an interpretability analysis was carried out for the obtained Random Forest model. The study was conducted in the Python with the use of relevant packages.
# </div>

# ## 2. Exploratory Data Analysis <a class="anchor" id="2"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# This chapter includes a description of the dataset. The distributions of individual variables were examined and the relationships between the outcome variable and other variables were verified. The dataset was examined based on the degree of balance for the outcome variable.
# </div>

# ### 2.1. Dataset description <a class="anchor" id="3"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The dataset comes from kaggle.com and describes the characteristics of patients who have (or have not) been diagnosed with hypertension. The outcome variable is a binary target variable, which describes whether or not a patient suffers from hypertension. There are 26 083 observations in the data set, together with the outcome variable there are 14 variables, which are as follows:
# </div>
# 
# - **target** – outcome variable (1 – hypertension, 0 – no hypertension),
# 
# 
# - *age* – patient's age,
# 
# 
# - *sex* – patient's gender (1 – male, 0 – female),
# 
# 
# - *cp* – chest pain type (categorical variable),
# 
# 
# - *trestbps* – resting blood pressure (mmHg),
# 
# 
# - *chol* – serum cholestoral (mg/dl),
# 
# 
# - *fbs* – patient's fasting blood sugar (1 – higher than 120mg/dl, 0 – lower),
# 
# 
# - *restecg* – resting ECG result (categorical variable),
# 
# 
# - *thalach* – maximum heart rate achieved,
# 
# 
# - *exang* – exercise induced angina (1 – yes, 0 – no),
# 
# 
# - *oldpeak* – ST depression on ECG chart,
# 
# 
# - *slope* – the slope of the peak exercise ST segment,
# 
# 
# - *ca* – number of major vessels (0–3) colored by flourosopy,
# 
# 
# - *thal* – myocardial cell test results during exercise (categorical variable).
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# At the beginning, data types for individual variables were verified and single missing values in the column describing patients' gender were filled with the mode.
# </div>

# In[1]:


#Import and preview the dataset
import pandas as pd
hyper = pd.read_csv("hypertension_data.csv")
hyper.head()


# In[2]:


#Impute missing values with the mode
mode_sex = hyper['sex'].mode().iloc[0]
hyper['sex'] = hyper['sex'].fillna(mode_sex)
hyper['sex'].value_counts()


# In[3]:


#Set types of variables
hyper['target'] = hyper['target'].astype('category')
hyper['sex'] = hyper['sex'].astype('category')
hyper['cp'] = hyper['cp'].astype('category')
hyper['fbs'] = hyper['fbs'].astype('category')
hyper['restecg'] = hyper['restecg'].astype('category')
hyper['exang'] = hyper['exang'].astype('category')
hyper['slope'] = hyper['slope'].astype('category')
hyper['ca'] = hyper['ca'].astype('category')
hyper['thal'] = hyper['thal'].astype('category')
hyper['trestbps'] = hyper['trestbps'].astype('int64')
hyper['chol'] = hyper['chol'].astype('int64')
hyper['thalach'] = hyper['thalach'].astype('int64')
hyper.info()


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The dataset was also examined in terms of potential outliers for the numerical variables - boxplots were created for each variable. It can be seen that there are some outlier values for the variables thalach and trestpbs. These objects, however should not be removed as for this type of a medical problem they can contribute important information to the ML models.
#  </div>

# In[4]:


#Boxplots for numerical variables
import seaborn as sns
import matplotlib.pyplot as plt

data = hyper[['thalach', 'trestbps', 'oldpeak', 'chol']]

sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6)) 
for i, column in enumerate(data.columns):
    sns.boxplot(x=data[column], ax=axes[i//2, i%2], orient="h", color="skyblue")
    axes[i//2, i%2].set_xlabel(column)

fig.suptitle("Boxplots for numerical variables", fontsize=18) 
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ### 2.2. EDA: relationships between variables <a class="anchor" id="4"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# Firstly, the level of balance of the dataset was assessed - the percentage of observations of the positive class, which are patients with hypertension, and the percentage of observations of the negative class, which consists of healthy people.
#  </div>

# In[5]:


#Target variable - pie chart
freq_table = pd.DataFrame(hyper['target'].value_counts()).reset_index()
freq_table.columns = ['target', 'Freq']
freq_table['Percentage'] = round(freq_table['Freq'] / freq_table['Freq'].sum() * 100, 1)

plt.figure(figsize=(8, 8))
colors = ['red', 'green']
wedges, _, autotexts = plt.pie(freq_table['Freq'], colors=colors, autopct='', textprops={'fontsize': 12})  # Usuwamy domyślne etykiety
for i, autotext in enumerate(autotexts):
    autotext.set_text(f"{freq_table['Freq'].values[i]} ({freq_table['Percentage'].values[i]}%)")
    autotext.set_fontsize(18)

plt.title('Outcome variable - level of balance', fontsize=24)
plt.legend(title='Hypertension', labels=['Yes', 'No'], fontsize=14)
plt.axis('equal')
plt.show()


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The number of positive class observations exceeds the number of negative class observations, the proportion of patients with hypertension is 54.7%, while the proportion of healthy patients is 45.3%. This set can be assessed as well balanced. In the next step, the distributions of the individual categorical variables by healthy and sick patients were examined - the relationships are shown in the column graphs.
# </div>

# In[6]:


#Column charts (1)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

#Variable SEX
sns.countplot(data=hyper, x='sex', hue='target', palette=['green', 'red'], ax=axes[0, 0])
axes[0, 0].set_title('Variable SEX', fontsize=16)
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Number of patients')
axes[0, 0].legend(title='Hypertension', labels=['No', 'Yes'], fontsize=10)
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Female', 'Male'])
axes[0, 0].tick_params(axis='both', labelsize=10)

#Variable EXANG
sns.countplot(data=hyper, x='exang', hue='target', palette=['green', 'red'], ax=axes[0, 1])
axes[0, 1].set_title('Variable EXANG', fontsize=16)
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('Number of patients')
axes[0, 1].legend(title='Hypertension', labels=['No', 'Yes'], fontsize=10)
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_xticklabels(['No', 'Yes'])
axes[0, 1].tick_params(axis='both', labelsize=10)

#Variable FBS
sns.countplot(data=hyper, x='fbs', hue='target', palette=['green', 'red'], ax=axes[1, 0])
axes[1, 0].set_title('Variable FBS', fontsize=16)
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('Number of patients')
axes[1, 0].legend(title='Hypertension', labels=['No', 'Yes'], fontsize=10)
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(['0', '1'])
axes[1, 0].tick_params(axis='both', labelsize=10)

#Variable RESTECG
sns.countplot(data=hyper, x='restecg', hue='target', palette=['green', 'red'], ax=axes[1, 1])
axes[1, 1].set_title('Variable RESTECG', fontsize=16)
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('Number of patients')
axes[1, 1].legend(title='Hypertension', labels=['No', 'Yes'], fontsize=10)
axes[1, 1].tick_params(axis='both', labelsize=10)
axes[1, 1].set_xticks([0, 1, 2])
axes[1, 1].set_xticklabels(['0', '1', '2'])
plt.tight_layout()
plt.show()


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The graph showing the distribution of the variable describing gender illustrates that in the dataset, there is a similar number of women and men. Among both men and women, hypertensive patients predominate. The next graph shows the distributions of the variable exang, which determines whether the patient is experiencing post-exercise angina. This ailment occurs among a minority of patients (8518 observations), for the remaining individuals it was not detected (17565 observations). Interestingly, among those who do not suffer from angina (exang = 0), the number of hypertensive patients doubles the number of healthy people. In contrast, for patients with angina, the number of healthy people (green column) exceeds the number of people with hypertension over three times.
# </div>
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The variable fbs describes the level of blood sugar concentration, if the concentration exceeds a value of 120 mg/dl, the variable takes the value 1, otherwise the value is 0. Measurements for the vast majority of patients do not exceed the cut-off value (22177 observations). In this group of patients, there is a preponderance of hypertensive patients. Those with blood sugar levels above 120 mg/dl show signs of presence of diabetes. They represent a minority (3906 observations), in this group of patients the number of people with hypertension is similar to the number of healthy ones.
#      </div>
#      
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The last graph shows the distributions of the restecg variable, which describes the resting ECG result. A value of zero for this variable means that, based on the ECG graph, no suspicious symptoms were detected - such a result was recorded among 12702 patients. Among these, there is a preponderance of patients who do not have hypertension. The other two values of the restecg variable correspond to the presence of disease symptoms based on the ECG chart. There are many observations in the dataset for which restecg = 1, these are patients who were were diagnosed with an inappropriate ST-segment inflection on the ECG graph. In this group of patients, hypertensive ones predominate. It can therefore be concluded that the ECG findings are associated with the outcome variable. The figure below illustrates the distributions of the other categorical variables in an analogous manner.
# </div>

# In[7]:


#Column charts (2)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

#Variable CP
sns.countplot(data=hyper, x='cp', hue='target', palette=['green', 'red'], ax=axes[0, 0])
axes[0, 0].set_title('Variable CP', fontsize=16)
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Number of patients')
axes[0, 0].legend(title='Hypertension', labels=['No', 'Yes'], fontsize=10)
axes[0, 0].set_xticks([0, 1, 2, 3])
axes[0, 0].set_xticklabels(['0', '1', '2', '3'])
axes[0, 0].tick_params(axis='both', labelsize=10)

#Variable SLOPE
sns.countplot(data=hyper, x='slope', hue='target', palette=['green', 'red'], ax=axes[0, 1])
axes[0, 1].set_title('Variable SLOPE', fontsize=16)
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('Number of patients')
axes[0, 1].legend(title='Hypertension', labels=['No', 'Yes'], fontsize=10)
axes[0, 1].set_xticks([0, 1, 2])
axes[0, 1].set_xticklabels(['0', '1', '2'])
axes[0, 1].tick_params(axis='both', labelsize=10)

#Variable CA
sns.countplot(data=hyper, x='ca', hue='target', palette=['green', 'red'], ax=axes[1, 0])
axes[1, 0].set_title('Variable CA', fontsize=16)
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('Number of patients')
axes[1, 0].legend(title='Hypertension', labels=['No', 'Yes'], fontsize=10)
axes[1, 0].set_xticks([0, 1, 2, 3, 4])
axes[1, 0].set_xticklabels(['0', '1', '2', '3', '4'])
axes[1, 0].tick_params(axis='both', labelsize=10)

#Variable THAL
sns.countplot(data=hyper, x='thal', hue='target', palette=['green', 'red'], ax=axes[1, 1])
axes[1, 1].set_title('Variable THAL', fontsize=16)
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('Number of patients')
axes[1, 1].legend(title='Hypertension', labels=['No', 'Yes'], fontsize=10)
axes[1, 0].set_xticklabels(['0', '1', '2', '3'])
axes[1, 1].tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.show()


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The cp variable describes the type of chest pain. If there is no pain, the variable takes the value zero. The group of patients who have no chest pain represents the majority - 12314 observations. In this group, the number of healthy people doubles the number of people with hypertension. The other values taken by this variable are:
# </div>
# 
# - 𝑐𝑝 = 1 - chest pain, typical angina, occurs among 4456,
# 
# 
# - 𝑐𝑝 = 2 - chest pain, atypical angina, occurs among 7392 patients,
# 
# 
# - 𝑐𝑝 = 3 - chest pain unrelated to angina, occurs among 1921 patients. 
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# For all three values, the number of patients with hypertension exceeds the number of healthy patients. The slope variable describes the slope of the ST segment on the ECG graph. It is assumed that the curve should be rising or have a steady progression. This variable takes the following values:
# </div>
# 
# - 𝑠𝑙𝑜𝑝𝑒 = 0 - upsloping, occurs in 1,826 individuals. In this group of patients are slightly outnumbered by healthy individuals,
# 
# 
# - 𝑠𝑙𝑜𝑝𝑒 = 1 - zero slope (flat), occurs in 11990 individuals. Healthy people almost doubles the number of people with hypertension,
# 
# 
# - 𝑠𝑙𝑜𝑝𝑒 = 2 - downsloping, occurs in 12267 individuals. The number of people with hypertension nearly three times higher.
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The ca variable describes the number of major blood organs for which fluoroscopy showed the presence of abnormalities. Among as many as 15146 patients, no abnormalities during fluoroscopy. Interestingly, in this group, the percentage of patients with hypertension. For those with abnormalities, majority of them do not have hypertension - the exception being a small group of people in whom abnormalities were detected for the four major blood organs. 
# </div>
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The thal variable describes the results of the myocardial cell test during exercise physical exercise. It takes on the following values:
# </div>
# 
# - 𝑡ℎ𝑎𝑙 = 1 - healthy myocardial cells, such a measurement was recorded for 1474 patients, with healthy subjects predominating among them,
# 
# 
# - 𝑡ℎ𝑎𝑙 = 2 - permanent irreversible abnormalities detected (14359 patients), the number of hypertensive patients is three times larger,
# 
# 
# - 𝑡ℎ𝑎𝑙 = 3 - detected reversible abnormalities (10096 patients), the proportion of people with hypertension is a distinct minority.
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# By reviewing the distributions of the individual variables for the positive and negative class, it is possible to conclude that most of the considered variables have significant influence on the outcome binary variable. 
#     </div>

# ## 3. ML algorithms for classification <a class="anchor" id="5"></a>
# <div style="text-align:justify;line-height: 1.6; margin-top:10px;">
# In order to predict a binary outcome variable that determines whether a patient shows signs of developing hypertension, models were created based on selected machine learning classification methods and their predictive ability was assessed based on relevant metrics. Before building the model, the data were divided into a training set and a test set. The learning set comprises 80% of the total dataset, while the test set is the remaining 20% - the division was performed randomly.
# </div>

# In[8]:


#Train test split
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(666)
train, test = train_test_split(hyper, test_size=0.2, stratify=hyper['target'])
X_train = train.drop('target', axis=1)
y_train = train['target']
X_test = test.drop('target', axis=1)
y_test = test['target']

#Numbers of objects in sets
train_target_counts = train['target'].value_counts()
test_target_counts = test['target'].value_counts()
print("Train set:")
print(train_target_counts)
print("\nTest set:")
print(test_target_counts)


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# As mentioned in the previous chapter, the dataset can be assessed as well balanced, the positive class is the majority class in this case, but the difference between the object counts of the two classes is not substantial. There is no need to apply resampling methods in the modelling process.
#     </div>
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# It should be emphasised that for classification issues in machine learning for medical problems, the most important element is the high quality of the classification of patients whose risk of disease - in this case, hypertension is high. Thus, in the evaluation process of the resulting classifier, special attention should be paid to the sensitivity coefficient, which is responsible for the classification quality of the objects from positive class.
#     </div>

# ### 3.1. Decision Tree <a class="anchor" id="6"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The first applied algorithm is Decision Tree. This algorithm requires tuning the values of several hyperparameters, which specify the level of complexity of a model. More complex decision tree enables the model to fit into the patterns in the training set more effectively but there is a risk of so called overfitting. Therefore, methods like grid search can be applied to check several values for existing hyperparameters in order to avoid underfitting and overfitting. The values for following hyperparameters were fixed:
# </div>
# 
# - *criterion*
# 
# 
# - *max-depth*
# 
# 
# - *min_samples_split*
# 
# 
# - *min_samples_leaf*

# In[9]:


#Decision Tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
pipeline = Pipeline([
    ('dt', DecisionTreeClassifier())
])

#Hyperparameters
param_grid = {
    'dt__criterion': ['gini', 'entropy'],
    'dt__max_depth': [None, 5, 10, 15],
    'dt__min_samples_split': [2, 5, 10],
    'dt__min_samples_leaf': [1, 2, 4]
}

model_dt = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
model_dt.fit(X_train, y_train)
print("Best values for hyperparameters:")
print(model_dt.best_params_)


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# Having built a Decision Tree model, the classification of patients from the test set was carried out. All patients from the test set were correctly assigned to their respective classes. Thus, a perfect classifier was obtained with 100% success rate. The confusion matrix for the Decision Tree model, which is shown below, confirms the obtained results.
# </div>

# In[10]:


#Decision Tree - confusion matrix
from sklearn.metrics import confusion_matrix

#Train test forecast
y_pred2 = model_dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree - confusion matrix')
plt.show()


# ### 3.2. Random Forest <a class="anchor" id="7"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# Another method used in the study is the Random Forest. This is a method of ensemble learning, involving the construction of multiple decision trees, where in each tree the splits are made based on a random subset of variables. Each tree makes its own prediction, then these predictions are combined according to the principle of majority voting. Random Forest models are more complex than single Decision Trees and they usually have more possibilities to generalize the new data. The main advantages of Random Forest models are stability and resistance to overfitting - thanks to the randomness in the construction of the trees, this algorithm tends to reduce the variance of the model. To build a suitable model, it is necessary to adjust the values of hyperparameters, in terms of Random Forests the most important ones are:
# </div>
# 
# - *n_estimators* - number of trees in the forest
# 
# 
# - *max_depth* - maximum depth of a model
# 
# 
# - *min_samples_split*

# In[11]:


#Random Forest Model
from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([
    ('rf', RandomForestClassifier())
])

#Hyperparameters
param_grid_rf = {
    'rf__n_estimators': [100],
    'rf__max_depth': [5, 10],
    'rf__min_samples_split': [2, 5],
}

model_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='accuracy')
model_rf.fit(X_train, y_train)
print("Best values for hyperparameters:")
print(model_rf.best_params_)


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# In the study, the Random Forest model consists of 100 decision trees. In the learning process, five-fold cross validation was used in order to avoid overfitting and tune hyperparameters. Having created the Random Forest model, the predictions were made for patients from the test set. Results are presented below in the confusion matrix.
#    </div>

# In[12]:


#Random Forest - confusion matrix
from sklearn.metrics import confusion_matrix
y_pred2 = model_rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest - confusion matrix')
plt.show()


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# Each patient from the test set was classified correctly again, the Random Forest model is also a perfect classifier. The result is as expected, as Random Forest is a more complex algorithm with greater predictive capabilities compared to single Decision Tree.  Obtaining a model that assigns objects from the test set to classes with 100% accuracy is rare - it depends largely on the quality of the data. The considered dataset is unlikely to describe the real data, it is probably a set created for educational purposes, the data are consistent, exemplary, do not contain outlier observations and do not require special complex preprocessing. In reality, the data is usually not of such a good quality and there is always some element of noise, so it is unrealistic to obtain a a perfect classifier.
#     </div>

# ### 3.3. SVM <a class="anchor" id="8"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The next algorithm used in the project is a Support Vector Machine (SVM) classifier. The essence of this method involves partitioning a multidimensional space using a hyperplane to separate the objects of the two classes. Partitioning is possible when the objects are linearly separable, otherwise, kernel functions are used to increase the number of dimensions of the
# data. When creating an SVM algorithm, the following hyperparameters need to be adjusted:
# </div>
# 
# - *Kernel function* - transformation of input data,
# 
# 
# - *C* - coefficient of regularization.

# In[13]:


#SVM model (linear kernel)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

X_train_subset = X_train.sample(n=3000, replace=False, random_state=42)
y_train_subset = y_train.sample(n=3000, replace=False, random_state=42)

#Scaling numerical variables
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_subset), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

#Hyperparameters
param_grid = {
    'C': np.linspace(0.1, 1, 10)
}

svm_classifier = SVC(kernel='linear', probability=True)
svm_model_linear = GridSearchCV(svm_classifier, param_grid, scoring='accuracy')
svm_model_linear.fit(X_train_scaled, y_train_subset)

print("Best values for hyperparameters:")
print(svm_model_linear.best_params_)


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# For educational purposes, the SVM model with linear kernel was built without using cross validation. The best value for regularization hyperparameter C is 1 which is the highest considered value. Such result is logical because the higher regularization parameter value, the more complex model we obtain. The training set was also limited to 3000 observations in order to decrease the duration of learning process. Including such restrictions, it is expected that the obtained model will not be able to predict the new data perfectly as the previously created models. The confusion matrix presenting the results of predictions for test set is shown below. The graph showing how the change in value of regularization parameter (C) was also presented. 
#  </div>

# In[14]:


#SVM linear - accuracy and C hyperparameter
results = pd.DataFrame(svm_model_linear.cv_results_)
param_values = results['param_C']
accuracy_values = results['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(param_values, accuracy_values, marker='o', linestyle='-', color='b')
plt.xlabel('C hyperparameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.title('SVM linear - accuracy and C hyperparameter', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# In[15]:


#SVM linear - confusion matrix
from sklearn.metrics import confusion_matrix
y_pred3 = svm_model_linear.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred3)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM linear - confusion matrix')
plt.show()


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# As expected, the obtained results of predictions for created SVM linear model are significantly worse. Not including cross validation in the learning process and restricting learning sample to 3000 observations had a negative impact on the quality of the model. Metrics such as accuracy, sensitivity and specificity were calculated to compare predictions in positive and negative class.
#     </div>

# In[16]:


#SVM linear - metrics of the model
from sklearn.metrics import accuracy_score, recall_score

accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
sensitivity =  cm[1, 1] / (cm[1, 0] + cm[1, 1])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print("Accuracy:    {:.2%}".format(accuracy))
print("Sensitivity: {:.2%}".format(sensitivity))
print("Specificity: {:.2%}".format(specificity))


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The strong point of this model is that the quality of classifying patients from positive class (people with hypertension) is clearly higher than the quality of predictions for healthy people. The sensitivity score is equal to 92.47% which means that on average, in the group of 10 people with hypertension, more than 9 of them are recognized as ill.
# </div>
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# In order to check the importance of kernel function hyperparameter, the other SVM model was created with Gaussian kernel function. Learning process was performed on the same subset of data as in SVM linear model. SVM model with Gausiann kernel requires fixing the value for hyperparameter gamma which controls the width of the Gaussian kernel. A small value of gamma indicates a wider kernel, which means the influence of a single training example will be more widespread. On the other hand, a large value of gamma creates a narrower kernel, leading to a more localized influence of a single training example.
#  </div>

# In[17]:


#SVM model (Gaussian kernel)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#Hyperparameters
param_grid = {
    'C': np.linspace(0.1, 1, 10),
    'gamma': np.array([0.1, 1, 10])
}

svm_classifier_rbf = SVC(kernel='rbf', probability=True)
svm_model_rbf = GridSearchCV(svm_classifier_rbf, param_grid, scoring='accuracy')
svm_model_rbf.fit(X_train_scaled, y_train_subset)

print("Best values for hyperparameters:")
print(svm_model_rbf.best_params_)


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# Line chart presented below shows how the hyperparameters value for C and gamma influence the accuracy of the model. Increase in the regularization rate C results in higher accuracy. For gamma parameter equal to 1, the accuracy rate is significantly the highest.
# </div>

# In[18]:


#SVM RBF kernel - accuracy and C hyperparameter
results = pd.DataFrame(svm_model_rbf.cv_results_)
param_values_C = results['param_C'].unique().astype('float64')
accuracy_values_gamma_0 = results[results['param_gamma'] == 0.1]['mean_test_score']
accuracy_values_gamma_1 = results[results['param_gamma'] == 1]['mean_test_score']
accuracy_values_gamma_2 = results[results['param_gamma'] == 10]['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(param_values_C, accuracy_values_gamma_0, marker='o', linestyle='-', color='b', label='Gamma = 0.1')
plt.plot(param_values_C, accuracy_values_gamma_1, marker='o', linestyle='-', color='g', label='Gamma = 1')
plt.plot(param_values_C, accuracy_values_gamma_2, marker='o', linestyle='-', color='r', label='Gamma = 10')
plt.xlabel('C hyperparameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.title('SVM with RBF kernel - accuracy and C hyperparameter', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# A confusion matrix was also created for the model to visualize the prediction quality on the test set.

# In[19]:


#SVM RBF - confusion matrix
from sklearn.metrics import confusion_matrix
y_pred4 = svm_model_rbf.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred4)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM RBF - confusion matrix')
plt.show()


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The predictive performance of the SVM model with a Gaussian kernel is significantly better compared to the SVM model with a linear kernel. Even with the limitations imposed by omitting cross-validation in the learning process, the model performed exceptionally well, correctly classifying almost all patients from the test set. Basic model metrics were also determined and presented below. Over 99% of patients were classified correctly.
# </div>

# In[20]:


#SVM RBF - metrics of the model
accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
sensitivity =  cm[1, 1] / (cm[1, 0] + cm[1, 1])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print("Accuracy:    {:.2%}".format(accuracy))
print("Sensitivity: {:.2%}".format(sensitivity))
print("Specificity: {:.2%}".format(specificity))


# ### 3.4. Bagging <a class="anchor" id="9"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The next applied algorithm is bagging. Similar to random forests, bagging is an ensemble learning algorithm where multiple individual base models are constructed, usually decision trees. The difference compared to random forests is that in bagging, each decision tree is trained on a different subset of the data. This is done through random sampling with replacement, meaning each observation can be selected more than once – this is called bootstrap sampling. After training all the decision tree models, the results are aggregated through majority voting. In bagging, the following hyperparameters need to be adjusted:
# </div>
# 
# - *n_estimators* - number of base models (decision trees),
# 
# 
# - *max_samples* - size of the sample used in the training process of an individual tree.

# In[21]:


#Bagging model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

base_model = DecisionTreeClassifier()  #base model - Decision Tree
bagging_model = BaggingClassifier(base_model, random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_samples': [0.5, 0.7]
}

bagging_model = GridSearchCV(bagging_model, param_grid, scoring='accuracy', cv=5)
bagging_model.fit(X_train, y_train)

print("Best hyperparameters:")
print(bagging_model.best_params_)


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# In the project, a model was built in which each tree is constructed using a sample containing half the number of observations compared to the entire training set (n_estimators = 0.5). Fifty decision trees were built in the model, and a five-fold cross-validation was applied during the training process. Predictions were made for the test set based on the constructed model. Confusion matrix for the bagging model is shown below.
# </div>

# In[22]:


#Bagging - confusion matrix
from sklearn.metrics import confusion_matrix
y_pred5 = bagging_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred5)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Bagging - confusion matrix')
plt.show()


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# Similarly to the random forests case, a perfect classifier was obtained, which correctly classifies all patients in the test set. All model performance measures are 100%.
#     </div>

# ### 3.5. Gradient Boosting <a class="anchor" id="10"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# Another approach to ensemble learning is based on boosting algorithms. The key difference between bagging and boosting lies in the fact that, in bagging, each simple model is trained independently, whereas in boosting, each subsequent simple model is trained sequentially. In the subsequent models, there are weights assigned to the observations; observations that were misclassified in previous models have higher weights and, therefore, are more likely to be selected in the next models. This way, subsequent base classifiers (decision trees) have to deal with predicting "harder" cases that were previously misclassified. Each subsequent tree tries to correct the errors of the previous trees.
# </div>
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# One of the most popular examples of boosting is Gradient Boosting. The name of the algorithm comes from the fact that the models are built based on the gradient (vector of derivatives) of the loss function. In the case of classification problems, the loss function can be cross-entropy. The Gradient Boosting algorithm works sequentially; a set of simple models (weak learners) like decision trees is constructed to create a strong classifier. The main idea is to minimize the loss function by iteratively adding subsequent models to the ensemble. Gradient Boosting requires tuning the following hyperparameters:
# </div>
# 
# - *n_estimators* - number of base models (weak learners - usually decision trees),
# 
# 
# - *max-depth* - maximum depth of the decision tree,
# 
# 
# - *learning_rate* - controls the contribution of each tree in the ensemble,
# 
# 
# - *min_samples_split* - minimum number of samples required to split a node.

# In[23]:


#Gradient Boosting model
from sklearn.ensemble import GradientBoostingClassifier

#Hyperparameters - grid search
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'min_samples_split': [5, 10, 15]
}

gb_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model = GridSearchCV(gb_model, param_grid, scoring='accuracy', cv=5)
gradient_boosting_model.fit(X_train, y_train)

print("Best hyperparameters:")
print(gradient_boosting_model.best_params_)


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The best Gradient Boosting model was obtained using 100 base models (decision trees). Maximum depth of the single decision tree in created model is equal to 4. Based on the results obtained for the previous models, it can be expected that the Gradient Boosting model will also be a perfect classifier. Confusion matrix presented below confirms that.
# </div>

# In[24]:


#Gradient Boosting - confusion matrix
from sklearn.metrics import confusion_matrix
y_pred6 = gradient_boosting_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred6)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Gradient Boosting - confusion matrix')
plt.show()


# ### 3.6. XGBoost <a class="anchor" id="11"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# The last of the applied algorithms is XGBoost (Extreme Gradient Boosting), which is an extension of the Gradient Boosting algorithm described in the previous section. XGBoost introduces more advanced optimization algorithms, utilizing gradient optimization methods that also consider the second derivatives of the loss function (Newton-Raphson method). Additionally, it incorporates L1 or L2 regularization techniques to prevent overfitting and improve the generalization of the model. XGBoost requires tuning the following hyperparameters:
# </div>
# 
# - *nrounds* - the number of iterations,
# 
# 
# - *max_depth* - the maximum depth of each tree in the ensemble,
# 
# 
# - *eta* - the learning rate,
# 
# 
# - *gamma* - the minimum loss reduction required to make further splits,
# 
# 
# - *colsample_bytree* - the fraction of features randomly sampled for building each tree,
# 
# 
# - *min_child_weight* - the minimum sum of instance weight,
# 
# 
# - *subsample* - the fraction of samples used for fitting the individual base learners.
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# It can be observed that the number of hyperparameters that need to be properly specified is quite large. In the project, the parameters responsible for L1 or L2 regularization were not taken into account because overfitting was not encountered so far. Cross-validation was also not considered during the training process to shorten the training time.
# </div>

# In[25]:


#XGBoost model
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

#Encoding categorical variables
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

#Create DMatrix with enable_categorical=True
dtrain = xgb.DMatrix(X_train_encoded, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test_encoded, label=y_test, enable_categorical=True)

#Hyperparameters
param_grid = {
    'max_depth': [3, 4, 5],
    'eta': [0.01, 0.1],
    'gamma': [0.1, 0.2],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [5, 10],
    'subsample': [0.6, 0.8]
}

xgb_model = xgb.XGBClassifier()
xgb_model_grid = GridSearchCV(xgb_model, param_grid, scoring='accuracy')
xgb_model_grid.fit(X_train_encoded, y_train)

print("Best hyperparameters:")
print(xgb_model_grid.best_params_)


# Confusion matrix for the XGBoost model with the best hyperparameters values is presented below.

# In[26]:


#XGBoost - confusion matrix
from sklearn.metrics import confusion_matrix
y_pred7 = xgb_model_grid.predict(X_test_encoded)

cm = confusion_matrix(y_test, y_pred7)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost - confusion matrix')
plt.show()


# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# All patients from the test set were classified correctly even despite not including cross validation in the learning process. The performance metrics of the model, such as accuracy, sensitivity, and specificity, all have a value of 100%. Considering that XGBoost is an advanced version of Gradient Boosting, the obtained result is not surprising.
# </div>

# ## 4. Conclusion <a class="anchor" id="12"></a>
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# In the project, the analysis was conducted to examine factors influencing the risk of developing arterial hypertension among patients. The preliminary analysis revealed dependencies between the considered variables and the outcome variable, as well as explored the distributions of the variables. The main part of the project involved utilizing selected machine learning classification algorithms to predict the binary outcome variable determining whether a patient has arterial hypertension. The following methods were employed: decision tree, SVM, Random Forest, Gradient Boosting, and XGBoost.
#     </div>
# 
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# For more complex algorithms such as Random Forest, bagging, Gradient Boosting, and XGBoost, after hyperparameter tuning, excellent classifiers were obtained, accurately assigning all objects from the test set to their respective classes. The surprising high effectiveness of classifiers may be attributed to the dataset being designed for educational purposes, characterized by consistency and homogeneity.
#     </div>
#     
# <div style="width: 90%; text-align:justify;line-height: 1.6; margin-top:10px;">
# For individual decision trees and SVM with a linear kernel, models with slightly weaker predictive abilities were achieved. Unfortunately, when obtaining multiple models with an accuracy of 100%, it becomes difficult to pinpoint the most efficient and effective algorithm. Recently, Gradient Boosting algorithms and their extensions (including XGBoost) have gained immense popularity, usually performing better with complex datasets when compared to Random Forests. However, for the considered dataset, utilizing Random Forest algorithm alone was sufficient to achieve a highly effective model
#     </div>
