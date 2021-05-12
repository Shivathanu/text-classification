#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pymysql
import itertools
import numpy as np

def dbConnect():
    myConnection = pymysql.connect( host="127.0.0.1", user="root", passwd="9840876156", db="promessa_jira" )
    return myConnection

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        element = element.split('|')
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def doQueryProj( conn ) :
    cur = conn.cursor()
    cur.execute( """
            SELECT * FROM promessa_jira.jira_issues_proj
            WHERE epic <> '';
        """
    )
    data=[]
    data = cur.fetchall()
    #print(data)

    conn.commit()
    conn.close()
    
    return data

def doQueryTask( conn ) :
    cur = conn.cursor()
    cur.execute( """
            SELECT id, summary, clean_summary AS clean_summary
            FROM promessa_jira.jira_issues_data;
        """
    )
    data=[]
    for id, summary, clean_summary in cur.fetchall():
        data.append(clean_summary)
        #print(id, summary, clean_summary)
    #print(data)

    conn.commit()
    conn.close()
    
    return data

def doQueryEpic( conn ) :
    cur = conn.cursor()
    cur.execute( """
            SELECT id, summary, clean_summary AS clean_summary
            FROM promessa_jira.jira_issues_data
            WHERE type='Epic';
        """
    )
    data=[]
    for id, summary, clean_summary in cur.fetchall():
        data.append(clean_summary)
        #print(id, summary, clean_summary)
    #print(data)

    conn.commit()
    conn.close()
    
    return data


# In[24]:


conn = dbConnect()
data = doQueryProj( conn )


# ### Data preparation ###
# 
# Have the project, epic and issue mapped together

# In[25]:


import pandas as pd
df = pd.DataFrame(data, columns=['project_id', 'project_key', 'project_name', 'summary', 'type', 'issue', 'epic', 'labels', 'created', 'clean_summary', 'project_status', 'epic_summary', 'epic_clean_summary'])

# print(df.head(10))
# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]


# ### Issues count per project ###
# 
# | project_id | project_key | project_name | task_count | epic_count
# | --- | --- | --- | --- | --- |
# | 109| EXLUEILWITZ| Rerum sequi non et.| 47| 6| 
# | 286| REICIENDISMACGYVER| Dolorem alias quis at perferendis.| 81| 2| 
# | 292| VELITONDRICKA| Modi facilis iste vel at aut.| 27| 3| 
# | 455| ETLANG| Consequatur unde optio autem.| 165| 28| 
# | 459| QUAEWATSICA| Provident animi atque facere minus.| 621| 12| 
# | 581| MOLLITIAWEHNER| Ut incidunt aliquid maxime voluptas eveniet laborum adipisci doloribus.| 78| 7| 
# | 616| OPTIOHERMANN-RUTHERFORD| Quia voluptatem in et.| 179| 15| 
# | 652| FACILISJAKUBOWSKI| Sit fuga et voluptatem fugiat reprehenderit quis enim omnis.| 72| 6| 
# | 688| ODIODURGAN-BOEHM| Eius nam omnis aut dolorem.| 156| 15| 
# | 798| ODIORUECKER-WATSICA| Aut autem nisi aut eaque et odio.| 22| 5| 
# | 818| CORPORISVEUM-HEATHCOTE| Esse pariatur fugiat enim est.| 63| 10| 
# | 840| NISIRUTHERFORD-TROMP| Explicabo aut expedita quod cupiditate quo sed maiores.| 271| 14| 
# | 938| FACILISLUBOWITZ| Quaerat porro nihil vel.| 6| 1| 
# | 950| CONSEQUATURCASSIN-GLOVER| Amet earum delectus quo.| 136| 8| 
# 
# 

# In[26]:


from io import StringIO

col = ['issue', 'epic', 'summary', 'clean_summary', 'epic_clean_summary', 'project_key']
df_encode = df[col]
df_encode = df_encode[pd.notnull(df_encode['summary'])]
df_encode.columns = ['issue', 'epic', 'summary', 'clean_summary', 'epic_clean_summary', 'project_key']
df['category_id'] = df['epic'].factorize()[0]
df_encode['category_id'] = df['category_id']
category_id_df = df[['epic', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'epic']].values)
df_encode['clean_summary'] = df_encode['clean_summary'].str.replace('|',' ')
df_encode['epic_clean_summary'] = df_encode['epic_clean_summary'].str.replace('|',' ')
# df['range'] = df['range'].str.replace(',','-')
df_encode


# In[27]:


# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(25,6))
# df_encode.groupby('epic').summary.count().plot.bar(ylim=0)
# plt.show()


# ### Get the count of the word in the text document using "CountVectorizer"

# In[28]:


from sklearn.feature_extraction.text import CountVectorizer

def get_count_vectorizer_matrix(proj_train):
    count_vect = CountVectorizer(binary=True)
    X_train_counts = count_vect.fit_transform(proj_train)
    return X_train_counts, count_vect


# ### Apply TF-IDF method to create document frequency score

# In[29]:


from sklearn.feature_extraction.text import TfidfTransformer

def get_tfidf_matrix(X_train_counts):
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
#     print(X_train_tfidf)
    return X_train_tfidf


# ### Predict the categories based on the transformed text vectors

# In[53]:


#docs_new = ['receive|message ', 'hour|message|wrong', 'offline|access|improvement', 'improvement', '']
from sklearn.feature_extraction.text import TfidfTransformer

def predict(model, count_vect, X_train_tfidf, y_train, X_test):
    clf = model.fit(X_train_tfidf, y_train)
    docs_new = X_test
    tfidf_transformer = TfidfTransformer()
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)
#     print('-----PREDICTION-----')
#     for doc, category in zip(docs_new, predicted):
#        print('%r => %s' % (doc, category))

    return predicted


# In[31]:


from sklearn.metrics import accuracy_score

def get_pred_score_half_split(df, proj, model):
    # get the train and test dataset for the projects

    X_train_df = df[(df.project_key == 'QUAEWATSICA') | (df.project_key == 'ETLANG') |
                  (df.project_key == 'VELITONDRICKA') | (df.project_key == 'MOLLITIAWEHNER') |
                  (df.project_key == 'EXLUEILWITZ') | (df.project_key == 'ODIODURGAN-BOEHM') |
                  (df.project_key == 'REICIENDISMACGYVER')]

    X_train = X_train_df['clean_summary']
    y_train = X_train_df['epic']

    X_test_df = df[(df.project_key == 'OPTIOHERMANN-RUTHERFORD') | 
                  (df.project_key == 'CONSEQUATURCASSIN-GLOVER') | (df.project_key == 'FACILISJAKUBOWSKI') |
                  (df.project_key == 'NISIRUTHERFORD-TROMP') | (df.project_key == 'ODIORUECKER-WATSICA') |
                  (df.project_key == 'CORPORISVEUM-HEATHCOTE') | (df.project_key == 'FACILISLUBOWITZ') | 
                  (df.project_key == 'ODIORUECKER-WATSICA')]

    X_test = X_test_df['clean_summary']
    y_test = X_test_df['epic']

    print('X_train::', X_train.shape)
    print('Y_train::', y_train.shape)
    print('X_test::', X_test.shape)
    print('Y_test::', y_test.shape)
    
    # count vectorizer
    count_vectorizer_output = get_count_vectorizer_matrix(X_train)
    X_train_counts = count_vectorizer_output[0]
    count_vect = count_vectorizer_output[1]
    
    # tfidf matrix
    X_train_tfidf = get_tfidf_matrix(X_train_counts)
    
    # model trained and predicted with the test data
    predicted = predict(model, count_vect, X_train_tfidf, y_train, X_test)
    
    return accuracy_score(y_test, predicted)
    
#     return np.mean(predicted == np.array(proj_test['clean_summary']))


# In[32]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_pred_score_train_test_split(df, proj, model):
    # get the train and test dataset for the projects

    X_train, X_test, y_train, y_test = train_test_split(df['clean_summary'], df['epic'], test_size=0.50, random_state = 0)

    print('X_train::', X_train.shape)
    print('Y_train::', y_train.shape)
    print('X_test::', X_test.shape)
    print('Y_test::', y_test.shape)
    
    # count vectorizer
    count_vectorizer_output = get_count_vectorizer_matrix(X_train)
    X_train_counts = count_vectorizer_output[0]
    count_vect = count_vectorizer_output[1]
    
    # tfidf matrix
    X_train_tfidf = get_tfidf_matrix(X_train_counts)
    
    # model trained and predicted with the test data
    predicted = predict(model, count_vect, X_train_tfidf, y_train, X_test)
    
    return accuracy_score(y_test, predicted)
    
#     return np.mean(predicted == np.array(proj_test['clean_summary']))


# ### Apply cross validation and get the scores

# In[33]:


from sklearn.metrics import accuracy_score

def get_cross_val_score(df, proj, model):
    # get the train and test dataset for the projects

    X_train_df = df.loc[df['project_key'] != proj] #Get the data other than the project
    X_train = X_train_df['clean_summary']
    y_train = X_train_df['epic']

    X_test_df = df.loc[df['project_key'] == proj] #Get the data only for that project
    X_test = X_test_df['clean_summary']
    y_test = X_test_df['epic']

    print('X_train::', X_train.shape)
    print('Y_train::', y_train.shape)
    print('X_test::', X_test.shape)
    print('Y_test::', y_test.shape)
    
    # count vectorizer
    count_vectorizer_output = get_count_vectorizer_matrix(X_train)
    X_train_counts = count_vectorizer_output[0]
    count_vect = count_vectorizer_output[1]
    
    # tfidf matrix
    X_train_tfidf = get_tfidf_matrix(X_train_counts)
    
    # model trained and predicted with the test data
    predicted = predict(model, count_vect, X_train_tfidf, y_train, X_test)
    
    return accuracy_score(y_test, predicted)
    
#     return np.mean(predicted == np.array(proj_test['clean_summary']))


# ### (Approach 1): Apply cross validation model

# In[34]:


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=450, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()

projects = ['QUAEWATSICA', 'ETLANG', 'VELITONDRICKA', 'MOLLITIAWEHNER', 'EXLUEILWITZ', 'ODIODURGAN-BOEHM', 'REICIENDISMACGYVER', 'OPTIOHERMANN-RUTHERFORD', 'ODIORUECKER-WATSICA', 'FACILISJAKUBOWSKI', 'NISIRUTHERFORD-TROMP', 'CORPORISVEUM-HEATHCOTE', 'FACILISLUBOWITZ', 'CONSEQUATURCASSIN-GLOVER']
model_scores = []

for proj in projects:
    print('project: %s' % proj)
    score = get_cross_val_score(df_encode, proj, MultinomialNB())
    print('model score for %s => %s' % (proj, score))
    model_scores.append(score)
    
print('list of model scores:', model_scores)
print('average score', np.array(model_scores).mean())


# ### (Approach 2): Split the projects into 50 percentage

# In[35]:


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=450, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()

get_pred_score_half_split(df_encode, '', LinearSVC())


# ### (Approach 3): Get train test split using sklearn model selection

# In[36]:


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()

get_pred_score_train_test_split(df_encode, '', SGDClassifier())


# ### (Approach 4): Apply cross validation using 6-fold split for tasks within a project

# In[95]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.feature_extraction.text import CountVectorizer

def get_cross_val_task(df, proj, model):
    # get the train and test dataset for the projects

    project_data = df.loc[df['project_key'] == proj]
    model_scores = []
    
    print('project_data.shape::', project_data.shape)

    X = project_data['clean_summary']
    y = project_data['epic']
    
    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=6, shuffle=True, random_state=1)
#     kf = StratifiedKFold(n_splits=6)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        count_vectorizer_output = get_count_vectorizer_matrix(X_train)
        X_train_counts = count_vectorizer_output[0]
        count_vect = count_vectorizer_output[1]
    
        # tfidf matrix
        X_train_tfidf = get_tfidf_matrix(X_train_counts)

        predicted = predict(model, count_vect, X_train_tfidf, y_train, X_test)
        
        return accuracy_score(y_test, predicted)
    


# In[103]:


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()

project_list = [
    'QUAEWATSICA', 
    'ETLANG', 
    'VELITONDRICKA', 
    'MOLLITIAWEHNER', 
    'EXLUEILWITZ', 
    'ODIODURGAN-BOEHM', 
    'REICIENDISMACGYVER', 
    'OPTIOHERMANN-RUTHERFORD', 
    'ODIORUECKER-WATSICA', 
    'NISIRUTHERFORD-TROMP', 
    'CORPORISVEUM-HEATHCOTE', 
    'CONSEQUATURCASSIN-GLOVER',
    'FACILISLUBOWITZ',
    'FACILISJAKUBOWSKI', 
]
model_scores = []

for proj in project_list:
    print('project: %s' % proj)
    score = get_cross_val_task(df_encode, proj, SGDClassifier())
    print('model score for %s => %s' % (proj, score))
    model_scores.append(score)
    
# print('list of model scores:', model_scores)
print('average score', np.array(model_scores).mean())


# In[ ]:




