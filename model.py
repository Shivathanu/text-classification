#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd

def readCSV():
    df = pd.read_csv('./jira_data.csv') 
    
    return df


# In[23]:


df = readCSV()


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

# ### Encode the data for classification performance

# In[15]:


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


# ### Get the count of the word in the text document using "CountVectorizer"

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer

def get_count_vectorizer_matrix(proj_train):
    count_vect = CountVectorizer(binary=True)
    X_train_counts = count_vect.fit_transform(proj_train)
    
    return X_train_counts, count_vect


# ### Apply TF-IDF method to create document frequency score

# In[17]:


from sklearn.feature_extraction.text import TfidfTransformer

def get_tfidf_matrix(X_train_counts):
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    return X_train_tfidf


# ### Predict the categories based on the transformed text vectors

# In[18]:


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


# ### Apply cross validation and get the scores

# In[11]:


from sklearn.metrics import accuracy_score

def get_cross_val_score(df, proj, model):
    # get the train and test dataset for the projects

    X_train_df = df.loc[df['project_key'] != proj] #Get the data other than the project
    X_train = X_train_df['clean_summary']
    y_train = X_train_df['epic']

    X_test_df = df.loc[df['project_key'] == proj] #Get the data only for that project
    X_test = X_test_df['clean_summary']
    y_test = X_test_df['epic']

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

# In[27]:


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=450, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()

projects = ['QUAEWATSICA', 'ETLANG', 'VELITONDRICKA', 'MOLLITIAWEHNER', 'EXLUEILWITZ', 'ODIODURGAN-BOEHM', 'REICIENDISMACGYVER', 'OPTIOHERMANN-RUTHERFORD', 'ODIORUECKER-WATSICA', 'FACILISJAKUBOWSKI', 'NISIRUTHERFORD-TROMP', 'CORPORISVEUM-HEATHCOTE', 'FACILISLUBOWITZ', 'CONSEQUATURCASSIN-GLOVER']
model_scores = []

for proj in projects:
    score = get_cross_val_score(df_encode, proj, MultinomialNB())
    print('model score for %s => %s' % (proj, score))
    model_scores.append(score)
    
print('list of model scores:', model_scores)
print('average score', np.array(model_scores).mean())


# ### (Approach 2): Split the projects into 50 percentage

# In[9]:


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


# In[64]:


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=450, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()

get_pred_score_half_split(df_encode, '', LinearSVC())


# ### (Approach 3): Get train test split using sklearn model selection

# In[10]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_pred_score_train_test_split(df, proj, model):
    # get the train and test dataset for the projects

    X_train, X_test, y_train, y_test = train_test_split(df['clean_summary'], df['epic'], test_size=0.50, random_state = 0)
    
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


# In[65]:


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()

get_pred_score_train_test_split(df_encode, '', SGDClassifier())


# ### (Approach 4): Apply cross validation using 6-fold split for tasks within a project

# In[34]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.pipeline import Pipeline

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
#     print('number of splitting iterations::', kf.get_n_splits(X, y))

    accuracy_scores = []
    weighted_accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        count_vectorizer_output = get_count_vectorizer_matrix(X_train)
        X_train_counts = count_vectorizer_output[0]
        count_vect = count_vectorizer_output[1]
    
        # tfidf matrix
        X_train_tfidf = get_tfidf_matrix(X_train_counts)

        predicted = predict(model, count_vect, X_train_tfidf, y_train, X_test)
        print('total instances:', len(X))
        print('no of instances used:', len(X_test))
        print('accuaracy of a model:', accuracy_score(y_test, predicted))
        weighted_score = (len(X_test) / len(X)) * accuracy_score(y_test, predicted)
        print('weighted_score accuracy of a model:', weighted_score)
        
        accuracy_scores.append(accuracy_score(y_test, predicted))
        weighted_accuracy_scores.append(weighted_score)
        f1_scores.append(f1_score(y_test, predicted, average='weighted'))
        precision_scores.append(precision_score(y_test, predicted, average='weighted'))
        recall_scores.append(recall_score(y_test, predicted, average='weighted'))
        
    print('accuracy for 6-splits:', accuracy_scores)
    print('f1_scores for 6-splits:', f1_scores)
    print('precision_scores for 6-splits:', precision_scores)
    print('recall_scores for 6-splits:', recall_scores)
    print('weighted_accuracy_scores for 6-splits:', weighted_accuracy_scores)
    
    return accuracy_scores, f1_scores, precision_scores, recall_scores, np.array(weighted_accuracy_scores).sum()


# In[35]:


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)
from sklearn.svm import SVC #SVM: LinearSVC()
from xgboost import XGBClassifier #XGBoost: XGBClassifier()

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
#     'FACILISLUBOWITZ',
#     'FACILISJAKUBOWSKI', 
]
model_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
weighted_accuracy_scores = []

for proj in project_list:
    print('project: %s' % proj)
    scores = get_cross_val_task(df_encode, proj, SVC(kernel='linear', gamma="auto"))
    print('mean model accuracy score for %s => %s' % (proj, np.array(scores[0]).mean()))
    print('mean model f1 score for %s => %s' % (proj, np.array(scores[1]).mean()))
    print('mean model precision score for %s => %s' % (proj, np.array(scores[2]).mean()))
    print('mean model recall score for %s => %s' % (proj, np.array(scores[3]).mean()))
    print('mean model weighted accuracy score for %s => %s' % (proj, np.array(scores[4]).mean()))
#     print('mean model roc auc score for %s => %s' % (proj, np.array(scores[4]).mean()))
    print('')
    model_scores.append(np.array(scores[0]).mean())
    f1_scores.append(np.array(scores[1]).mean())
    precision_scores.append(np.array(scores[2]).mean())
    recall_scores.append(np.array(scores[3]).mean())
    weighted_accuracy_scores.append(np.array(scores[4]).mean())

    
    
print('list of model scores:', model_scores)
print('average model accuracy score', np.array(model_scores).mean())
print('average model f1 score', np.array(f1_scores).mean())
print('average model precision score', np.array(precision_scores).mean())
print('average model recall score', np.array(recall_scores).mean())
print('average model weighted accuracy score', np.array(weighted_accuracy_scores).mean())


# ### (Approach 5): Apply Optuna hyper parameter tuning

# In[47]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_cross_val_task_optuna(trial):
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
#         'FACILISLUBOWITZ',
#         'FACILISJAKUBOWSKI', 
    ]
    
    accuracy_scores = []
    f1_scores = []
    
    for proj in project_list:
        f1_scores_split = []
        print('project: %s' % proj)
        df = df_encode
        project_data = df.loc[df['project_key'] == proj]

        print('project_data.shape::', project_data.shape)

        X = project_data['clean_summary']
        y = project_data['epic']

        X = np.array(X)
        y = np.array(y)
        kf = KFold(n_splits=6, shuffle=True, random_state=1)

# #         Model 1: Random Forest        
#         n_estimators = trial.suggest_int('n_estimators', 2, 20)
#         max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)


# #         Model 2: LinearSVC
#         model = SVC(kernel='linear', gamma="auto")

# #         Model 3: XGBoost
        param = {
                    "n_estimators" : trial.suggest_int('n_estimators', 0, 1000),
                    'max_depth':trial.suggest_int('max_depth', 2, 25),
                    'reg_alpha':trial.suggest_int('reg_alpha', 0, 5),
                    'reg_lambda':trial.suggest_int('reg_lambda', 0, 5),
                    'min_child_weight':trial.suggest_int('min_child_weight', 0, 5),
                    'gamma':trial.suggest_int('gamma', 0, 5),
                    'learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5),
                    'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01),
                    'nthread' : -1
                }
        model = XGBClassifier(**param)


# #         Model 4: SVC
#          svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
#          model = sklearn.svm.SVC(C=svc_c, gamma='auto')

        
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            count_vectorizer_output = get_count_vectorizer_matrix(X_train)
            X_train_counts = count_vectorizer_output[0]
            count_vect = count_vectorizer_output[1]

            # tfidf matrix
            X_train_tfidf = get_tfidf_matrix(X_train_counts)

            predicted = predict(model, count_vect, X_train_tfidf, y_train, X_test)
            f1_scores_split.append(f1_score(y_test, predicted, average='weighted'))
    
        f1_scores.append(np.array(f1_scores_split).mean())

    print('f1_scores:', np.array(f1_scores).mean())
    return np.array(f1_scores).mean()


# ### Get the optimal f1-score with 100 trials

# In[48]:


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()
from xgboost import XGBClassifier #XGBoost: XGBClassifier()
import optuna

study = optuna.create_study(direction='maximize')
study.optimize(get_cross_val_task_optuna, n_trials=100)

trial = study.best_trial

print('F1 Score: {}'.format(trial.value))


# ### Results of the models: ###
# 
# ### Metrics used Accuracy, F1-score, Precision, Recall, Weighted Average
# 
# | --- | Random forest |Multinomial Naive Bayes|SGDClassifier|LinearSVC|XGBoost
# | --- | --- | --- | --- | --- | --- |
# | Accuracy| 50.46746988| 55.80569899| 61.08376065| 62.07692338| 56.64141844
# | F1-Score| 40.82062274| 48.73799776| 60.29856176| 57.50378251| 52.16391074
# | Precision| 39.25255127| 48.89874888| 63.55695146| 58.65400480| 53.1796943
# | Recall| 50.46746988| 55.80569899| 61.08376065| 62.07692338| 56.64141844
# | Weighted accuracy| 50.52285601| 55.82911205| 61.69389651| 62.19266598| 56.76536158

# ### F1 Scores optimized with optuna
# 
# Random Forest: 
# Trial 99 finished with value: 0.5422728892031464 and parameters: {'n_estimators': 17, 'max_depth': 18.694494544210215}. Best is trial 71 with value: 0.56837380556999.
# 
# SVC:
# Trial 99 finished with value: 60.29856176 and parameters: {'n_estimators': 17, 'max_depth': 18.694494544210215}. Best is trial 0 with value: 0.6029856176.
# 
# LinearSVC:
# Trial 99 finished with value: 0.5750378251934886 and parameters: {'n_estimators': 9, 'max_depth': 1.8792925549037873}. Best is trial 0 with value: 0.5750378251934886.
# 
# XGBoost:
# Trial 99 finished with value: 0.3480685389875591 and parameters: {'n_estimators': 643, 'max_depth': 22, 'reg_alpha': 0, 'reg_lambda': 2, 'min_child_weight': 4, 'gamma': 0, 'learning_rate': 0.032516760757620125, 'colsample_bytree': 0.86}. Best is trial 97 with value: 0.5618991864631359.

# In[ ]:





# In[ ]:




