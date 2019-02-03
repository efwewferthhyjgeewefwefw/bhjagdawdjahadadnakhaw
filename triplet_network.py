
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
def random_select_ids(num, id_list):
    random_list = []
    for i in range(num):
        random_list.append(random.choice(id_list))
    return random_list

# pair ids, 3 parts: input_ids，pos_ids，neg_ids
def create_pair_ids(df, label_name, target):
    num = 0
    pos_df = df[df[label_name] == 1]
    neg_df = df[df[label_name] == 0]  
    pos_num = pos_df.shape[0]
    neg_num = neg_df.shape[0]
    pos_id_list = pos_df['id'].tolist()
    neg_id_list = neg_df['id'].tolist()
    
    input_ids = []
    pos_ids = []
    neg_ids = []
    
    while num < target:
        # create pos pair
        for i in range(0, pos_num):
            if (num >= target):
                break  
            input_id = pos_id_list[i]
            pair_num = (target - num) if (target - num) < (pos_num - 1) else (pos_num - 1)
            
            input_ids = input_ids + [input_id] * pair_num
            pos_id_candidate = pos_df[pos_df['id']!=input_id]['id'].tolist()
            pos_ids = pos_ids + pos_id_candidate[:pair_num]
            neg_ids = neg_ids + random_select_ids(pair_num, neg_id_list)
            num += pair_num
            
        for j in range(0, neg_num):
            if (num >= target):
                break    
            input_id = neg_id_list[j]
            pair_num = (target - num) if (target - num) < (neg_num - 1) else (neg_num - 1)
            
            input_ids = input_ids + [input_id] * pair_num
            pos_id_candidate = neg_df[neg_df['id']!=input_id]['id'].tolist()
            pos_ids = pos_ids + pos_id_candidate[:pair_num]
            neg_ids = neg_ids + random_select_ids(pair_num, pos_id_list)
            num += pair_num
        
    return input_ids, pos_ids, neg_ids

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
pair_save_dir = './data/hotel_pair_data/triplet'

hotel_train = pd.read_csv(os.path.join(hotel_data_dir, 'train.csv'))
hotel_val = pd.read_csv(os.path.join(hotel_data_dir, 'validation.csv'))
hotel_test = pd.read_csv(os.path.join(hotel_data_dir, 'test.csv'))

print('original train size: ', hotel_train.shape[0])
print('original val size: ', hotel_val.shape[0])
print('original test size: ', hotel_test.shape[0])

input_ids, pos_ids, neg_ids = create_pair_ids(hotel_train, 'ground_truth', int(1e5))
random.seed(42)
random.shuffle(input_ids)
random.seed(42)
random.shuffle(pos_ids)
random.seed(42)
random.shuffle(neg_ids)
print(len(input_ids))
print(len(pos_ids))
print(len(neg_ids))
create_pairs(input_ids, hotel_train, 'id', pair_save_dir, 'train_query.csv')
create_pairs(pos_ids, hotel_train, 'id', pair_save_dir, 'train_pos.csv')
create_pairs(neg_ids, hotel_train, 'id', pair_save_dir, 'train_neg.csv')

input_ids, pos_ids, neg_ids = create_pair_ids(hotel_val, 'ground_truth', int(2e4))
random.seed(42)
random.shuffle(input_ids)
random.seed(42)
random.shuffle(pos_ids)
random.seed(42)
random.shuffle(neg_ids)
print(len(input_ids))
print(len(pos_ids))
print(len(neg_ids))
create_pairs(input_ids, hotel_val, 'id', pair_save_dir, 'val_query.csv')
create_pairs(pos_ids, hotel_val, 'id', pair_save_dir, 'val_pos.csv')
create_pairs(neg_ids, hotel_val, 'id', pair_save_dir, 'val_neg.csv')


# *******************  2. train triplet network *********************
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def triplet_loss(y_true, y_pred):
    margin = K.constant(0.2)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return K.l2_normalize(x, axis=-1)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    # x = Flatten()(input)
    x = Dense(128, activation='relu')(input)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    return Model(input, x)


# import data
hotel_dir = '/media/data/fuweiping/paper/AIED2019/triplet/data/hotel'
hotel_query_tr = pd.read_csv(os.path.join(pair_save_dir, 'train_query.csv'))
hotel_pos_tr = pd.read_csv(os.path.join(pair_save_dir, 'train_pos.csv'))
hotel_neg_tr = pd.read_csv(os.path.join(pair_save_dir, 'train_neg.csv'))
hotel_query_val = pd.read_csv(os.path.join(pair_save_dir, 'val_query.csv'))
hotel_pos_val = pd.read_csv(os.path.join(pair_save_dir, 'val_pos.csv'))
hotel_neg_val = pd.read_csv(os.path.join(pair_save_dir, 'val_neg.csv'))

input_shape = (hotel_query_tr.shape[1],)

# network definition
base_network = create_base_network(input_shape)

input_q = Input(shape=input_shape)
input_p = Input(shape=input_shape)
input_n = Input(shape=input_shape)

processed_q = base_network(input_q)
processed_p = base_network(input_p)
processed_n = base_network(input_n)

pos_dist = Lambda(euclidean_distance, name='pos_dist')([processed_q, processed_p])
neg_dist = Lambda(euclidean_distance, name='neg_dist')([processed_q, processed_n])

combine_dists = Lambda(
    lambda vects: K.stack(vects, axis=1), 
    name='combine_dists')([pos_dist, neg_dist])

h_model = Model([input_q, input_p, input_n], combine_dists)

# rms = RMSprop()
sgd = SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

h_model.compile(loss=triplet_loss,
             optimizer=sgd,
             metrics=[accuracy])

y_tr = np.empty((hotel_query_tr.shape[0],))
y_val = np.empty((hotel_query_val.shape[0],))
h_model.fit([hotel_query_tr, hotel_pos_tr, hotel_neg_tr], y_tr,
        epochs=500,
        batch_size=256,
        validation_data=([hotel_query_val, hotel_pos_val, hotel_neg_val], y_val),
        callbacks=[earlystop, checkpoint])

# save final model
h_model.save('hotel_triplet_model.h5')


# *******************  3. predict by LR *********************
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

# net_X_tr = np.float64(net_X_tr)
# net_X_val = np.float64(net_X_val)
# net_X_ts = np.float64(net_X_ts)

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

# - 3层dense + dropout*2：0.1，learning rate：0.005，epoch：500，batch size：256
# 
# - accuracy 0.8571428571428571
# - precision 0.8472222222222222
# - recall 0.8714285714285714
# - f1 0.8591549295774648
# - auc 0.926734693877551

# save embeddings
hotel_emd_savedir = './embeddings/triplet'
emd_to_csv(hotel_emd_savedir, 'net_X_tr.csv', h_net_X_tr)
emd_to_csv(hotel_emd_savedir, 'net_X_val.csv', h_net_X_val)
emd_to_csv(hotel_emd_savedir, 'net_X_ts.csv', h_net_X_ts)
