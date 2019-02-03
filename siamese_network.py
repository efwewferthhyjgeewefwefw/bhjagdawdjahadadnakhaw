
# coding: utf-8

import os
import pandas as pd
import numpy as np
import keras
import csv
import random
import keras_metrics
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,f1_score,precision_score



# *******************  1. create pair data  *********************
# pair ids, 3 parts: input1_ids，input2_ids，y
def create_pair_ids(df, label_name, target):
    length = df.shape[0]
    input1_ids = []
    input2_ids = []
    y = []
    cur_num = 0
    num = length
    for i in range(length):
        num = length-i-1 if (target - cur_num) >= (length-i-1) else (target - cur_num)
        input1_id = df.iloc[i]['id']
        label = df.iloc[i][label_name] 
        
        input1_ids += [input1_id]*num
        input2_ids += df[i+1:]['id'].tolist()[:num]
        all_labels = [1 if x else 0 for x in list(df[label_name] == label)]
        labels = all_labels[i+1:]
        y += labels[:num]
        cur_num += num
         
        if cur_num >= target:
            break
            
    return input1_ids, input2_ids, y

# create pairs according to ids
def create_pairs(id_list, df, join_label, save_dir, filename):
    length = len(id_list)
    
    with open (os.path.join(save_dir, filename), 'a') as output:
        csv_writer = csv.writer(output, delimiter=',')
        columns = list(df)
        csv_writer.writerow(columns)
   
        for i in range(0, length):
            arr = df[df[join_label]==id_list[i]][columns].values[0]
            csv_writer.writerow(arr)


hotel_data_dir = './data/hotel_origin_data'
pair_save_dir = './data/hotel_pair_data/siamese'

hotel_train = pd.read_csv(os.path.join(hotel_data_dir, 'train.csv'))
hotel_val = pd.read_csv(os.path.join(hotel_data_dir, 'validation.csv'))
hotel_test = pd.read_csv(os.path.join(hotel_data_dir, 'test.csv'))

print('original train size: ', hotel_train.shape[0])
print('original val size: ', hotel_val.shape[0])
print('original test size: ', hotel_test.shape[0])

# create hotel train pairs
input1_ids, input2_ids, y = create_pair_ids(hotel_train, 'ground_truth', int(1e5))

random.seed(42)
random.shuffle(input1_ids)
random.seed(42)
random.shuffle(input2_ids)
random.seed(42)
random.shuffle(y)

print(len(input1_ids))
print(len(input2_ids))
print(len(y))

create_pairs(input1_ids, hotel_train, 'id', pair_save_dir, 'train_input1.csv')
create_pairs(input2_ids, hotel_train, 'id', pair_save_dir, 'train_input2.csv')
df = pd.DataFrame({'input1_id': input1_ids, 'input2_id': input2_ids, 'label': y})
df.to_csv(os.path.join(save_dir, 'train_label.csv'))

# create hotel validation pairs
input1_ids, input2_ids, y = create_pair_ids(hotel_val, 'ground_truth', int(2e4))

random.seed(42)
random.shuffle(input1_ids)
random.seed(42)
random.shuffle(input2_ids)
random.seed(42)
random.shuffle(y)

print(len(input1_ids))
print(len(input2_ids))
print(len(y))

create_pairs(input1_ids, hotel_val, 'id', pair_save_dir, 'val_input1.csv')
create_pairs(input2_ids, hotel_val, 'id', pair_save_dir, 'val_input2.csv')
df = pd.DataFrame({'input1_id': input1_ids, 'input2_id': input2_ids, 'label': y})
df.to_csv(os.path.join(save_dir, 'val_label.csv'))



# *******************  2. train siamese network *********************
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Dense(128, activation='relu')(input)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu', name='embedding')(x)
    return Model(input, x)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# start training siamese network
# import pair data
hotel_tr1 = pd.read_csv(os.path.join(pair_save_dir, 'train_input1.csv'))
hotel_tr2 = pd.read_csv(os.path.join(pair_save_dir, 'train_input2.csv'))
hotel_tr_y = pd.read_csv(os.path.join(pair_save_dir, 'train_label.csv'))
hotel_val1 = pd.read_csv(os.path.join(pair_save_dir, 'val_input1.csv'))
hotel_val2 = pd.read_csv(os.path.join(pair_save_dir, 'val_input2.csv'))
hotel_val_y = pd.read_csv(os.path.join(pair_save_dir, 'val_label.csv'))

# handle input data
hotel_cols = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'votes', 'ground_truth']
hotel_X_tr1 = hotel_tr1.drop(columns=hotel_cols, axis=1)
hotel_X_tr2 = hotel_tr2.drop(columns=hotel_cols, axis=1)
hotel_y_tr = hotel_tr_y['label']
hotel_X_val1 = hotel_val1.drop(columns=hotel_cols, axis=1)
hotel_X_val2 = hotel_val2.drop(columns=hotel_cols, axis=1)
hotel_y_val = hotel_val_y['label']


input_shape = (hotel_X_tr1.shape[1],)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

h_model = Model([input_a, input_b], distance)

# rms = RMSprop()
sgd = SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
checkpoint = ModelCheckpoint('hotel_weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

h_model.compile(loss=contrastive_loss,
            optimizer=sgd)
h_model.fit([hotel_X_tr1, hotel_X_tr2], hotel_y_tr,
        epochs=500,
        batch_size=128,
        validation_data=([hotel_X_val1, hotel_X_val2], hotel_y_val),
        callbacks=[earlystop, checkpoint])

# compute final accuracy on training and validation sets
y_pred = h_model.predict([hotel_X_tr1, hotel_X_tr2])
tr_acc = compute_accuracy(hotel_y_tr, y_pred)
y_pred = h_model.predict([hotel_X_val1, hotel_X_val2])
val_acc = compute_accuracy(hotel_y_val, y_pred)

print('* Accuracy on training set: %0.4f%%' % (100 * tr_acc))
print('* Accuracy on val set: %0.4f%%' % (100 * val_acc))

# save final model
h_model.save('hotel_siamese_model.h5')


# *******************  3. predict by LR *********************

# save model embeddings to csv
def emd_to_csv(savedir, filename, arr):
    df = pd.DataFrame(data=arr)
    df.to_csv(os.path.join(savedir, filename), mode='a', index=True)

# get first layer embedding
def get_emd(data, w, b):
    W = np.vstack([w, b])
    ones = np.ones((len(data),1))
    X = np.append(data, ones, axis=1)
    return np.dot(X, W)


# read original data
tr = pd.read_csv(os.path.join(hotel_data_dir, 'train.csv'))
val = pd.read_csv(os.path.join(hotel_data_dir, 'validation.csv'))
ts = pd.read_csv(os.path.join(hotel_data_dir, 'test.csv'))

h_X_tr = tr.drop(columns = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'votes', 'ground_truth'])
h_X_val = val.drop(columns = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'votes', 'ground_truth'])
h_X_ts = ts.drop(columns = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'votes', 'ground_truth'])
h_y_tr = tr['ground_truth']
h_y_val = val['ground_truth']
h_y_ts = ts['ground_truth']

# get embeddings
h_net_X_tr = base_network.predict(h_X_tr)
h_net_X_val = base_network.predict(h_X_val)
h_net_X_ts = base_network.predict(h_X_ts)

# LR grid search
penalty = ['l1', 'l2']
C = [0.001, 0.01, 0.1, 1, 10, 100, 100]

best_params = {'penalty': 'l1', 'C': 0.01, 'max_iter': 200}
best_acc = 0
i = 0
for l in penalty:
    for c in C:
        lr = LogisticRegression(penalty=l, C=c, max_iter=200)
        lr.fit(h_net_X_tr, h_y_tr)
        y_hat_tr = lr.predict(h_net_X_tr)
        y_hat_val = lr.predict(h_net_X_val)
        acc_tr = accuracy_score(y_hat_tr, h_y_tr)
        acc_val = accuracy_score(y_hat_val, h_y_val)
        # print(i)
        # i += 1
        if acc_val > best_acc:
            best_acc = acc_val
            best_params['penalty'] = l
            best_params['C'] = c
print('val acc: ', best_acc)

best_lr = LogisticRegression(penalty=best_params['penalty'], C=best_params['C'], max_iter=200)
best_lr.fit(h_net_X_tr, h_y_tr)
# y_hat_ts = best_lr.predict(h_net_X_ts)
# acc_ts = accuracy_score(y_hat_ts, h_y_ts)
# print('test acc: ', acc_ts)
y_hat_ts = best_lr.predict(h_net_X_ts)
y_proba_ts = best_lr.predict_proba(h_net_X_ts)

acc_ts = accuracy_score(y_hat_ts, h_y_ts)
precision = precision_score(h_y_ts, y_hat_ts)
recall = recall_score(h_y_ts, y_hat_ts)
f1 = f1_score(h_y_ts, y_hat_ts)
auc = roc_auc_score(h_y_ts, y_proba_ts[:,1])

print('accuracy', acc_ts)
print('precision', precision)
print('recall', recall)
print('f1', f1)
print('auc', auc)


# best model

# - 3 dense + dropout*2：0.1，learning rate：0.005，epoch：500，batch size：128
# 
# - accuracy 0.8642857142857143
# - precision 0.84
# - recall 0.9
# - f1 0.8689655172413793
# - auc 0.919795918367347

# save embeddings
hotel_emd_savedir = './embeddings/siamese'
emd_to_csv(hotel_emd_savedir, 'net_X_tr.csv', h_net_X_tr)
emd_to_csv(hotel_emd_savedir, 'net_X_val.csv', h_net_X_val)
emd_to_csv(hotel_emd_savedir, 'net_X_ts.csv', h_net_X_ts)
