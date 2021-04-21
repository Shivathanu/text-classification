#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


conn = dbConnect()
data = doQueryProj( conn )


# ### Data preparation ###
#
# Have the project, epic and issue mapped together

# In[3]:


import pandas as pd
df = pd.DataFrame(data, columns=['project_id', 'project_key', 'project_name', 'summary', 'type', 'issue', 'epic', 'labels', 'created', 'clean_summary', 'project_status', 'epic_summary', 'epic_clean_summary'])

print(df.head(10))

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

# ### Get the count of the word in the text document using "CountVectorizer"

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer

def get_count_vectorizer_matrix(proj_train):
    count_vect = CountVectorizer(binary=True)
    X_train_counts = count_vect.fit_transform(np.array(proj_train['clean_summary']))
    return X_train_counts, count_vect


# ### Apply TF-IDF method to create document frequency score

# In[5]:


from sklearn.feature_extraction.text import TfidfTransformer

def get_tfidf_matrix(X_train_counts):
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf


# ### Predict the categories based on the transformed text vectors

# In[15]:


#docs_new = ['receive|message ', 'hour|message|wrong', 'offline|access|improvement', 'improvement', '']
from sklearn.feature_extraction.text import TfidfTransformer

def predict(model, count_vect, X_train_tfidf, proj_train, proj_test):
    clf = model.fit(X_train_tfidf, np.array(proj_train['clean_summary']))
    docs_new = np.array(proj_test['clean_summary'])
    tfidf_transformer = TfidfTransformer()
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)
#     for doc, category in zip(docs_new, predicted):
#        print('%r => %s' % (doc, category))

    return predicted
#   for proj in projects:
#      print('%r => %s' % (doc, category))


# ### Apply cross validation and get the scores

# In[16]:


def get_cross_val_score(df, proj, model):
    # get the train and test dataset for the projects
    proj_train = df
    proj_train_df = df.loc[df['project_key'] != proj] #Get the data other than the projects
    proj_test = df.loc[df['project_key'] == proj] #Get the data only for that project

    # count vectorizer
    count_vectorizer_output = get_count_vectorizer_matrix(proj_train)
    X_train_counts = count_vectorizer_output[0]
    count_vect = count_vectorizer_output[1]

    # tfidf matrix
    X_train_tfidf = get_tfidf_matrix(X_train_counts)

    # model trained and predicted with the test data
    predicted = predict(model, count_vect, X_train_tfidf, proj_train, proj_test)

    return np.mean(predicted == np.array(proj_test['clean_summary']))


# In[17]:


from sklearn.linear_model import SGDClassifier #SVM
from sklearn.naive_bayes import MultinomialNB #MultinomialNB

projects = ['QUAEWATSICA', 'ETLANG', 'VELITONDRICKA', 'MOLLITIAWEHNER', 'EXLUEILWITZ', 'ODIODURGAN-BOEHM', 'REICIENDISMACGYVER', 'OPTIOHERMANN-RUTHERFORD', 'ODIORUECKER-WATSICA', 'FACILISJAKUBOWSKI', 'NISIRUTHERFORD-TROMP', 'CORPORISVEUM-HEATHCOTE', 'FACILISLUBOWITZ', 'CONSEQUATURCASSIN-GLOVER']
model_scores = []

for proj in projects:
    score = get_cross_val_score(df, proj, MultinomialNB())
    print('model score for %s => %s' % (proj, score))
    model_scores.append(score)

print('list of model scores:', model_scores)
print('average score', np.array(model_scores).mean())
