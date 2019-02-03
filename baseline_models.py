
# coding: utf-8

import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,f1_score,precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras.optimizers import SGD
from keras import backend as K
import sklearn.metrics as sklm
import keras_metrics


# hotel data
hotel_data_dir = './data/hotel_origin_data'
tr = pd.read_csv(os.path.join(hotel_data_dir, 'train.csv'))
val = pd.read_csv(os.path.join(hotel_data_dir, 'validation.csv'))
ts = pd.read_csv(os.path.join(hotel_data_dir, 'test.csv'))

X_tr = tr.drop(columns = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'votes', 'ground_truth'])
X_val = val.drop(columns = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'votes', 'ground_truth'])
X_ts = ts.drop(columns = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'votes', 'ground_truth'])
y_tr = tr['ground_truth']
y_val = val['ground_truth']
y_ts = ts['ground_truth']


# *******************  1. predict by LR *********************
penalty = ['l1', 'l2']
C = [0.001, 0.01, 0.1, 1, 10, 100, 100]

best_params = {'penalty': 'l1', 'C': 0.01, 'max_iter': 200}
best_acc = 0
i = 0
for l in penalty:
    for c in C:
        lr = LogisticRegression(penalty=l, C=c, max_iter=200)
        lr.fit(X_tr, y_tr)
        y_hat_tr = lr.predict(X_tr)
        y_hat_val = lr.predict(X_val)
        acc_tr = accuracy_score(y_hat_tr, y_tr)
        acc_val = accuracy_score(y_hat_val, y_val)
#         print(i)
#         i += 1
        if acc_val > best_acc:
            best_acc = acc_val
            best_params['penalty'] = l
            best_params['C'] = c
print('val acc: ', best_acc)

best_lr = LogisticRegression(penalty=best_params['penalty'], C=best_params['C'], max_iter=200)
best_lr.fit(X_tr, y_tr)
# y_hat_ts = best_lr.predict(X_ts)
# acc_ts = accuracy_score(y_hat_ts, y_ts)
# print('test acc: ', acc_ts)
y_hat_ts = best_lr.predict(X_ts)
y_proba_ts = best_lr.predict_proba(X_ts)

acc_ts = accuracy_score(y_hat_ts, y_ts)
precision = precision_score(y_ts, y_hat_ts)
recall = recall_score(y_ts, y_hat_ts)
f1 = f1_score(y_ts, y_hat_ts)
auc = roc_auc_score(y_ts, y_proba_ts[:,1])
conf_matrix = confusion_matrix(y_ts, y_hat_ts)

print('LR result:')
print('accuracy', acc_ts)
print('precision', precision)
print('recall', recall)
print('f1', f1)
print('auc', auc)


# *******************  2. predict by GBDT *********************
n_estimators = [5,10,30,50,100,200] 
learning_rate = [0.001,0.01,0.1,1] 
subsample = [0.5,0.7,0.9,1.0]  
max_depth = [1, 3, 5] 
max_features = [None, 'auto', 'log2']

best_n_estimators = 50
best_le = 1
best_subsample = 1.0
best_max_depth = 3
best_max_features = 'auto'

best_acc_val = 0
best_acc_tr = 0

for n in n_estimators:
    for l in learning_rate:
        for s in subsample:
            for d in max_depth:
                for f in max_features:
                    gbdt = GradientBoostingClassifier(n_estimators=n, learning_rate=l, subsample=s, max_depth=d, max_features=f, random_state=42)
                    gbdt.fit(X_tr, y_tr)
                    
                    y_hat_tr = gbdt.predict(X_tr)
                    y_hat_val = gbdt.predict(X_val)
                    acc_tr = accuracy_score(y_hat_tr, y_tr)
                    acc_val = accuracy_score(y_hat_val, y_val)
                    # print(acc_val)
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        best_acc_tr = acc_tr
                        best_n_estimators = n
                        best_le = l
                        best_subsample = s
                        best_max_depth = d
                        best_max_features = f


print('train acc {}'.format(best_acc_tr))
print('validation acc {}'.format(best_acc_val))
print('GBDT params: ')
print(best_n_estimators)
print(best_le)
print(best_subsample)
print(best_max_depth)
print(best_max_features)

best_gbdt = GradientBoostingClassifier(n_estimators=50, learning_rate=1, subsample=1.0, max_depth=3, random_state=42)
best_gbdt.fit(X_tr, y_tr)
y_hat_ts = best_gbdt.predict(X_ts)
y_proba_ts = best_gbdt.predict_proba(X_ts)

acc_ts = accuracy_score(y_hat_ts, y_ts)
precision = precision_score(y_ts, y_hat_ts)
recall = recall_score(y_ts, y_hat_ts)
f1 = f1_score(y_ts, y_hat_ts)
auc = roc_auc_score(y_ts, y_proba_ts[:,1])

print('GBDT result:')
print('accuracy', acc_ts)
print('precision', precision)
print('recall', recall)
print('f1', f1)
print('auc', auc)


# *******************  3. predict by DNN *********************
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# model definition
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=300))
# model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)

precision = keras_metrics.precision()
recall = keras_metrics.recall()
f1 = keras_metrics.f1_score()

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy', precision, recall, f1, auc])
model.fit(X_tr, y_tr,
          epochs=500,
          batch_size=64)

val_score = model.evaluate(X_val, y_val, batch_size=64)
print(val_score[1:])
test_score = model.evaluate(X_ts, y_ts, batch_size=64)
print('DNN result:')
print(test_score[1:])
