import pickle
import pandas as pd
from scipy import sparse
import collections
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import tensorflow.contrib.layers as layers




def load_neuronet(path, model):
    with tf.device('cpu:0'):
        tf.reset_default_graph()
        restore_graph = tf.Graph()
        w, b ={}, {}
        with tf.Session(graph=restore_graph) as restore_sess:
            restore_saver = tf.train.import_meta_graph(os.path.join(path, model))
            restore_saver.restore(restore_sess, tf.train.latest_checkpoint(path))
            g = tf.get_default_graph()
            fc_l1_query_w = g.get_tensor_by_name('fc_l1_query/weights:0').eval()
            fc_l1_query_b = g.get_tensor_by_name('fc_l1_query/biases:0').eval()
            fc_l1_doc_w = g.get_tensor_by_name('fc_l1_doc/weights:0').eval()
            fc_l1_doc_b = g.get_tensor_by_name('fc_l1_doc/biases:0').eval()

            fc_l2_query_w = g.get_tensor_by_name('fc_l2_query/weights:0').eval()
            fc_l2_query_b = g.get_tensor_by_name('fc_l2_query/biases:0').eval()
            fc_l2_doc_w = g.get_tensor_by_name('fc_l2_doc/weights:0').eval()
            fc_l2_doc_b = g.get_tensor_by_name('fc_l2_doc/biases:0').eval()
            w['fc_l1_query_w'] = fc_l1_query_w
            w['fc_l1_doc_w'] = fc_l1_doc_w
            w['fc_l2_query_w'] = fc_l2_query_w
            w['fc_l2_doc_w'] = fc_l2_doc_w

            b['fc_l1_query_b'] = fc_l1_query_b
            b['fc_l1_doc_b'] = fc_l1_doc_b
            b['fc_l2_query_b'] = fc_l2_query_b
            b['fc_l2_doc_b'] = fc_l2_doc_b
            return w, b
    
def getSigmoid(x):
    return 1.0/(1+np.exp(-x))

def forward_prop(w, b, x, which):
    if(which=='query'):
        l1_out = getSigmoid(np.dot(x, w['fc_l1_query_w'])+b['fc_l1_query_b'])
        return getSigmoid(np.dot(l1_out, w['fc_l2_query_w'])+b['fc_l2_query_b'])
    elif(which =='doc'):
        l1_out = getSigmoid(np.dot(x, w['fc_l1_doc_w'])+b['fc_l1_doc_b'])
        return getSigmoid(np.dot(l1_out, w['fc_l2_doc_w'])+b['fc_l2_doc_b'])



def input_data(which, train_path, val_path, test_path, w, b):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    #train = pd.concat([train, val], axis=0)
    test = pd.read_csv(test_path)
    X_train = train.drop(columns = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'ground_truth','votes'])
    print('feature dimension is {}'.format(X_train.shape[1]))
    y_train = train['ground_truth']
    X_val = val.drop(columns = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'ground_truth','votes'])
    y_val = val['ground_truth']
    X_test = test.drop(columns = ['id', 'mv_fluency', 'GLAD_label', 'p0', 'p1', 'ground_truth','votes'])
    y_test = test['ground_truth']
    if(which=='query'):
        X_train_dssm = forward_prop(w, b, X_train, 'query')
        X_val_dssm = forward_prop(w, b, X_val, 'query')
        X_test_dssm = forward_prop(w, b, X_test, 'query')
    elif(which=='doc'):
        X_train_dssm = forward_prop(w, b, X_train, 'doc')
        X_val_dssm = forward_prop(w, b, X_val, 'doc')
        X_test_dssm = forward_prop(w, b, X_test, 'doc')
    elif(which=='both'):
        X_train_dssm = np.concatenate((forward_prop(w, b, X_train, 'query'), forward_prop(w, b, X_train, 'doc')), axis=1)
        X_val_dssm = np.concatenate((forward_prop(w, b, X_val, 'query'), forward_prop(w, b, X_val, 'doc')), axis=1)
        X_test_dssm = np.concatenate((forward_prop(w, b, X_test, 'query'), forward_prop(w, b, X_test, 'doc')), axis=1)
    return X_train_dssm, y_train, X_val_dssm, y_val, X_test_dssm, y_test
