import pandas as pd
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import tensorflow.contrib.layers as layers
import os
from sklearn.utils import shuffle
import os
import scipy.stats
from utils import *
from sklearn.metrics import make_scorer,accuracy_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LogisticRegression
from scipy import spatial

plt.switch_backend('agg')


    
def getBatch(data, bs, idx):
    batch_dict = []
    for k, v in data.items():
        batch_dict.append(v[bs*idx : bs*(idx+1)])
    return batch_dict

def read_input_feature(fileList, feature_path, weight_path):
    data = {}
    for f in fileList:
        if(os.path.exists(os.path.join(feature_path, f))):
            df = pd.read_csv(os.path.join(feature_path, f))
            data[f[:-4]] = np.array(df, dtype=np.float32)

        if(os.path.exists(os.path.join(weight_path, f))):
            df = pd.read_csv(os.path.join(weight_path, f))['votes']
            #df = pd.read_csv(os.path.join(weight_path, f))['weight'] #for GLAD
            data[f[:-4]] = np.array(df, dtype=np.float32)
    return data

def cos_sim(a, b):
    normalize_a = tf.nn.l2_normalize(a,1)
    normalize_b = tf.nn.l2_normalize(b,1)
    cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b), axis=1)
    return cos_similarity

def cos_sim_1D(a, b):
    result = 1 - spatial.distance.cosine(a, b)
    return result


def plot_loss(tr_loss, val_loss, ts_loss, path, name):
    fig = plt.figure(1, (30, 18)) 
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(axis = 'x', which = 'major', labelsize = 50) 
    ax.tick_params(axis = 'y', which = 'major', labelsize = 50)
    plt.plot(tr_loss, label ='tr acc', marker = 's', markersize = 5, linewidth=5)
    plt.plot(val_loss, label ='val acc', marker = 's', markersize = 5, linewidth=5)
    plt.plot(ts_loss, label ='ts acc', marker = 's', markersize = 5, linewidth=5)


    plt.xlabel('iteration', fontsize=50)
    plt.ylabel('accuracy', fontsize=50)
    plt.legend(loc='lower right', fontsize=50)
    plt.savefig(os.path.join(path, name))
    plt.close()



def findTrainInstance_cosine(X_mistake, X_tr):
    target= []
    for i in range(X_mistake.shape[0]):
        x = np.array(X_mistake.iloc[i]).reshape(-1,)
        lst = []
        for j in range(X_tr.shape[0]):
            y = np.array(X_tr.iloc[j]).reshape(-1,)
            lst.append(cos_sim_1D(x, y))
        target.append(X_tr.iloc[np.argmax(lst)].values.reshape(-1))#for cosine we find largest cosine value
    return target



def enumerateAll(pos_ins, neg_ins, old_train):
    p_num, n_num = len(pos_ins), len(neg_ins)
    query_pdoc = [(x, y) for x in range(p_num) for y in range(p_num) if x!=y]
    ndoc = [(x, y, z) for x in range(n_num) for y in range(n_num) for z in range(n_num) if (x<y and y<z)]
    pairs = [(a,b,c,d,e) for (a,b) in query_pdoc for (c,d,e) in ndoc]
    
    pairs = shuffle(pairs)

    print('pairs size is {}'.format(len(pairs)))
    
    query = pd.DataFrame([pos_ins[x[0]] for x in pairs])
    p_d = pd.DataFrame([pos_ins[x[1]] for x in pairs])
    nd0 = pd.DataFrame([neg_ins[x[2]] for x in pairs])
    nd1 = pd.DataFrame([neg_ins[x[3]] for x in pairs])
    nd2 = pd.DataFrame([neg_ins[x[4]] for x in pairs])

    train_data = {}
    print('saving query')
    query_full = pd.concat([query, pd.DataFrame(old_train['query_train'])], axis=0)
    print('saving pos doc')
    pd_full = pd.concat([p_d, pd.DataFrame(old_train['pos_doc_train'])], axis=0)
    print('saving neg doc1')
    nd0_full = pd.concat([nd0, pd.DataFrame(old_train['neg_doc0_train'])], axis=0)
    print('saving neg doc2')
    nd1_full = pd.concat([nd1, pd.DataFrame(old_train['neg_doc1_train'])], axis=0)
    print('saving neg doc3')
    nd2_full = pd.concat([nd2, pd.DataFrame(old_train['neg_doc2_train'])], axis=0)
    train_data['query_train']=query_full
    train_data['pos_doc_train'] = pd_full
    train_data['neg_doc0_train'] = nd0_full
    train_data['neg_doc1_train'] = nd1_full
    train_data['neg_doc2_train'] = nd2_full
    return train_data

def getMetaName(path):
    return [x for x in os.listdir(path) if x[-4:]=='meta'][0]

def dssm_cosine(flag, train_data, test_data, bs, lr_rate, l1_n, l2_n, max_iter, reg_scale, dropout_rate, gamma, weighted, save_path, model_name,curve_path, trial):
    tf.reset_default_graph()
    dimension = train_data['query_train'].shape[1]
    is_training=tf.placeholder_with_default(False, shape=(), name='is_training')
    queryBatch = tf.placeholder(tf.float32, shape=[None, dimension], name='queryBatch')
    posDocBatch = tf.placeholder(tf.float32, shape=[None, dimension], name='posDocBatch')
    negDocBatch0 = tf.placeholder(tf.float32, shape=[None, dimension], name='negDocBatch0')
    negDocBatch1 = tf.placeholder(tf.float32, shape=[None, dimension], name='negDocBatch1')
    negDocBatch2 = tf.placeholder(tf.float32, shape=[None, dimension], name='negDocBatch2')

    with tf.name_scope('fc_l1_query'):
        query_l1_out = tf.contrib.layers.fully_connected(queryBatch, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                             activation_fn = tf.nn.sigmoid, scope='fc_l1_query')
        query_l1_out=tf.layers.dropout(query_l1_out, dropout_rate, training=is_training)
    with tf.name_scope('fc_l1_doc'):
        pos_doc_l1_out =  tf.contrib.layers.fully_connected(posDocBatch, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                            activation_fn = tf.nn.sigmoid, scope='fc_l1_doc')
        pos_doc_l1_out=tf.layers.dropout(pos_doc_l1_out, dropout_rate, training=is_training)

        neg_doc0_l1_out =  tf.contrib.layers.fully_connected(negDocBatch0, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                             activation_fn = tf.nn.sigmoid, scope='fc_l1_doc', reuse=True)
        neg_doc0_l1_out=tf.layers.dropout(neg_doc0_l1_out, dropout_rate, training=is_training)

        neg_doc1_l1_out =  tf.contrib.layers.fully_connected(negDocBatch1, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                              activation_fn = tf.nn.sigmoid, scope='fc_l1_doc', reuse=True)
        neg_doc1_l1_out=tf.layers.dropout(neg_doc1_l1_out, dropout_rate, training=is_training)

        neg_doc2_l1_out =  tf.contrib.layers.fully_connected(negDocBatch2, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                              activation_fn = tf.nn.sigmoid, scope='fc_l1_doc', reuse=True)

        neg_doc2_l1_out=tf.layers.dropout(neg_doc2_l1_out, dropout_rate, training=is_training)


    with tf.name_scope('fc_l2_query'):
        query_l2_out = tf.contrib.layers.fully_connected(query_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                         activation_fn = tf.nn.sigmoid, scope='fc_l2_query')
        query_l2_out=tf.layers.dropout(query_l2_out, dropout_rate, training=is_training)
    with tf.name_scope('fc_l2_doc'):
        pos_doc_l2_out = tf.contrib.layers.fully_connected(pos_doc_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),
                                                           activation_fn = tf.nn.sigmoid, scope='fc_l2_doc')
        pos_doc_l2_out=tf.layers.dropout(pos_doc_l2_out, dropout_rate, training=is_training)

        neg_doc0_l2_out = tf.contrib.layers.fully_connected(neg_doc0_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                            activation_fn = tf.nn.sigmoid, scope='fc_l2_doc', reuse=True)
        neg_doc0_l2_out=tf.layers.dropout(neg_doc0_l2_out, dropout_rate, training=is_training)

        neg_doc1_l2_out = tf.contrib.layers.fully_connected(neg_doc1_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                            activation_fn = tf.nn.sigmoid, scope='fc_l2_doc', reuse=True)
        neg_doc1_l2_out=tf.layers.dropout(neg_doc1_l2_out, dropout_rate, training=is_training)

        neg_doc2_l2_out = tf.contrib.layers.fully_connected(neg_doc2_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                            activation_fn = tf.nn.sigmoid, scope='fc_l2_doc', reuse=True)

        neg_doc2_l2_out=tf.layers.dropout(neg_doc2_l2_out, dropout_rate, training=is_training)

    with tf.name_scope('loss'):
        reg_ws_0 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_query')
        reg_ws_1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_doc')
        reg_ws_2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_query')
        reg_ws_3 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_doc')
        reg_loss = tf.reduce_sum(reg_ws_0)+tf.reduce_sum(reg_ws_1)+tf.reduce_sum(reg_ws_2)+tf.reduce_sum(reg_ws_3)

        nominator = tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, pos_doc_l2_out)))
        doc0_similarity = tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, neg_doc0_l2_out)))
        doc1_similarity = tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, neg_doc1_l2_out)))
        doc2_similarity = tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, neg_doc2_l2_out)))
        prob = tf.add(nominator, tf.constant(1e-10))/tf.add(doc0_similarity+ doc1_similarity+doc2_similarity+nominator,tf.constant(1e-10))
        log_prob = tf.log(prob)
        loss_batch = -tf.reduce_sum(log_prob) + reg_loss

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdadeltaOptimizer(lr_rate).minimize(loss_batch)




    train_size = train_data['query_train'].shape[0]
    test_size = test_data['query_test'].shape[0]

    print('training data size ', train_size)
    print('testing data size', test_size)

    best_test_loss = 2147483647
    best_train_loss = best_test_loss
    num_batch = train_size//bs
    saver = tf.train.Saver(max_to_keep=1,save_relative_paths=True)
    tr_acc_lst, val_acc_lst, ts_acc_lst = [], [], []

    tr_loss, ts_loss = [], []
    y_hat_tr_lst, y_hat_val_lst, y_hat_ts_lst = [], [], [] 
    #params_lst = []
    old_train = train_data

    train_path = '../data/train.csv'
    val_path = '../data/validation.csv'
    test_path = '../data/test.csv'

    start = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(max_iter):

            total_loss = 0
            for batch in range(0, num_batch):
                query_batch, pos_doc_batch, neg_doc0_batch, neg_doc1_batch, \
                        neg_doc2_batch = getBatch(train_data, bs, batch)
                feed = {    is_training: True,\
                            queryBatch: query_batch, \
                            posDocBatch : pos_doc_batch, \
                            negDocBatch0 :neg_doc0_batch, \
                            negDocBatch1 : neg_doc1_batch, \
                            negDocBatch2: neg_doc2_batch,\
                }
                _, batch_loss = sess.run([optimizer, loss_batch], feed_dict = feed)
                total_loss += batch_loss

            print("Epoch {} train loss {}, avg train loss {}".format(epoch, total_loss, total_loss/train_size))
            tr_loss.append(total_loss/train_size)

            query_ts, pos_doc_ts, neg_doc0_ts, neg_doc1_ts, neg_doc2_ts = getBatch(test_data, test_size, 0)
            feed_ts = {
                        is_training: False,
                        queryBatch: query_ts,
                        posDocBatch : pos_doc_ts,
                        negDocBatch0 :neg_doc0_ts,
                        negDocBatch1 : neg_doc1_ts,
                        negDocBatch2: neg_doc2_ts,
                        }

            test_loss = loss_batch.eval(feed_dict = feed_ts)
            ts_loss.append(test_loss/test_size)


            saver.save(sess, os.path.join(save_path, model_name), global_step=epoch)

            if(test_loss < best_test_loss):

                best_test_loss = test_loss
                best_train_loss = total_loss/train_size

            m = getMetaName(save_path)
            print('loading model from iteration {}'.format(epoch))
            w, b = load_neuronet(save_path, m)
            X_train_dssm, y_train, X_val_dssm, y_val, X_test_dssm, y_test = input_data('doc',train_path,val_path,test_path,w,b)


            model = LogisticRegression()
            model.fit(X_train_dssm, y_train)

            y_hat_tr = model.predict(X_train_dssm)
            y_hat_tr_lst.append(y_hat_tr)
            y_hat_val = model.predict(X_val_dssm)
            y_hat_val_lst.append(y_hat_val)
            y_hat_ts = model.predict(X_test_dssm)
            y_hat_ts_lst.append(y_hat_ts)

            accuracy_tr = accuracy_score(y_train, y_hat_tr)
            accuracy_val = accuracy_score(y_val, y_hat_val)
            accuracy_ts = accuracy_score(y_test, y_hat_ts)

            print('train accuracy {}'.format(accuracy_tr))
            print('val accuracy {}'.format(accuracy_val))
            print('test accuracy {}'.format(accuracy_ts))
            print('*'*60)
            tr_acc_lst.append(accuracy_tr)
            val_acc_lst.append(accuracy_val)
            ts_acc_lst.append(accuracy_ts)

            if(flag and accuracy_val>=0.75):
                train = pd.read_csv(train_path)
                val = pd.read_csv(val_path)
                mistake = val[[x!=y for x,y in zip(y_val, y_hat_val)]]
                pos_mistake = mistake[mistake['mv_fluency']==1]
                neg_mistake = mistake[mistake['mv_fluency']==0]

                X_tr = train.drop(columns = ['id', 'mv_fluency','GLAD_label', 'p0','p1', 'votes','ground_truth'])
                X_pos_mistake = pos_mistake.drop(columns = ['id', 'mv_fluency','GLAD_label', 'p0','p1', 'votes','ground_truth'])
                X_neg_mistake = neg_mistake.drop(columns = ['id', 'mv_fluency','GLAD_label', 'p0','p1', 'votes','ground_truth'])

                tr_pos_ins=findTrainInstance_cosine(X_pos_mistake, X_tr)
                tr_neg_ins = findTrainInstance_cosine(X_neg_mistake, X_tr)

                train_data = enumerateAll(tr_pos_ins, tr_neg_ins, old_train)

            if(not os.path.exists(curve_path)):
                os.makedirs(curve_path)
            plot_loss(tr_acc_lst, val_acc_lst, ts_acc_lst, curve_path, 'acc_{}.png'.format(trial))

            df1=pd.DataFrame(tr_acc_lst, columns=['tr_acc'])
            df2=pd.DataFrame(val_acc_lst, columns= ['val_acc'])
            df3=pd.DataFrame(ts_acc_lst, columns= ['ts_acc'])
            df=pd.concat([df1,df2, df3], axis=1)           
            df.to_csv(os.path.join(curve_path, 'acc_{}.csv'.format(trial)), index=False) 


            pd.DataFrame(y_hat_tr_lst).to_csv('/media/data/Guowei/paper/AIED2019/hotel/result/cosine/prediction/y_hat_tr_{}.csv'.format(trial), index=False)
            pd.DataFrame(y_hat_val_lst).to_csv('/media/data/Guowei/paper/AIED2019/hotel/result/cosine/prediction/y_hat_val_{}.csv'.format(trial), index=False)
            pd.DataFrame(y_hat_ts_lst).to_csv('/media/data/Guowei/paper/AIED2019/hotel/result/cosine/prediction/y_hat_ts_{}.csv'.format(trial), index=False)