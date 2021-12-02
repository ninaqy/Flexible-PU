import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

import utils.tf_util_1 as tf_util
from utils.tf_util_2 import feature_extraction
from tf_ops.grouping.tf_grouping import group_point 


def placeholder(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, None, 6))
    pointclouds_radius = tf.placeholder(tf.float32, shape=(batch_size))
    up_ratio = tf.placeholder(tf.int32, shape=(), name="model_up_ratio")
    is_training_pl = tf.placeholder(tf.bool, shape=())

    return pointclouds_pl, pointclouds_gt, pointclouds_radius, up_ratio, is_training_pl

def get_model(point_cloud, is_training,  up_ratio, max_ratio, bradius=1.0,k=30, topk=4,scope='generator', weight_decay=0, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    is_dist = False 
    bn = True
    bottleneck_size = 128

    batch_size = point_cloud.get_shape()[0].value
    num_point_sparse = point_cloud.get_shape()[1].value #n
    num_point_max = num_point_sparse*max_ratio # n*r_max
    num_point_up = num_point_sparse*up_ratio # n*r

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        adj_point = tf_util.pairwise_distance(point_cloud) #[b,n,n]
        features = feature_extraction(point_cloud, scope='feature_extraction', is_training=is_training, bn_decay=None) #[b,n,1,d]

        net = tf_util.conv2d(features, bottleneck_size, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                    bn=True, is_training=is_training, scope='concat_conv1', weight_decay=weight_decay, is_dist=is_dist) #[b,n,1,d]
 

        # obtain knn information
        knn_idx = tf_util.knn(adj_point, k=topk)  #[b,n,k] as index
        edge_feature = tf_util.get_edge_feature(net, nn_idx=knn_idx, k=topk) # [b,n,k,128*2]

        knn_point = group_point(point_cloud, knn_idx) # [b,n,k,3]
        xyz_tile = tf.tile(tf.expand_dims(point_cloud, axis=2), [1,1,topk,1]) # [b,n,k,3]

        dist_point = knn_point - xyz_tile  # [b,n,k,3]
        dist = tf.norm(dist_point, axis=-1, keepdims=True)  # [b,n,k,1]

        dist_feat = tf.concat([xyz_tile, knn_point, dist_point, dist], axis=-1)  # [b,n,k,d]

        dist_feat = tf_util.conv2d(dist_feat, bottleneck_size, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                    bn=True, is_training=is_training, scope='concat_conv2', weight_decay=weight_decay, is_dist=is_dist) # [b,n,k,d]

        # regress score
        feature_append = tf.concat([edge_feature, dist_feat], axis=-1) # [b,n,k,d]

        score_full = tf_util.conv2d(feature_append, 64, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='score_conv1', bn_decay=bn_decay, is_dist=is_dist)
        score_full = tf_util.conv2d(score_full, 64, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='score_conv2', bn_decay=bn_decay, is_dist=is_dist)

        score_full = tf_util.conv2d(score_full, max_ratio, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='score_conv3', bn_decay=bn_decay, is_dist=is_dist, activation_fn=None, bias=False) # [b,n,k,r_max]

        score_full = tf.nn.softmax(score_full, axis=2) # [b,n,k,r_max]
        score_full = tf.transpose(score_full, perm=[0,1,3,2]) # [b,n,r_max,k]
        score = score_full[:,:,0:up_ratio,:] # [b,n,r,k]
 
        # regress linear combination points
        S_point1_tmp = tf.matmul(score, knn_point)  # [b,n,r,3]
        S_point1 = tf.reshape(S_point1_tmp, [batch_size, num_point_up,3])  # [b,n*r,3]

        
        # self attention
        S_feat = tf.matmul(score, feature_append) # [b,n,r, 128*3]
        S_feat = tf.concat([S_point1_tmp, S_feat], axis=-1) # [b,n,r, 128*3]
        S_feat = tf.reshape(S_feat, [batch_size, num_point_up, 387]) # [b,n*r, 128*3]
        S_feat = tf_util.conv1d(S_feat, 128, 1,
                           padding='VALID', stride=1,
                           bn=False, is_training=is_training, weight_decay=weight_decay,
                           scope='sa_conv0', bn_decay=bn_decay, is_dist=is_dist)
        
        feat_q = tf_util.conv1d(S_feat, 128, 1,
                           padding='VALID', stride=1,
                           bn=False, is_training=is_training, weight_decay=weight_decay,
                           scope='sa_q1', bn_decay=bn_decay, is_dist=is_dist, bias=False) # [b,n*r, 128]
        feat_v = tf_util.conv1d(S_feat, 128, 1,
                           padding='VALID', stride=1,
                           bn=False, is_training=is_training, weight_decay=weight_decay,
                           scope='sa_v1', bn_decay=bn_decay, is_dist=is_dist) # [b,n*r, 128]
        energy = tf.matmul(feat_q, tf.transpose(feat_q, perm=[0,2,1])) #[b,n*r,n*r]
        attention = tf.nn.softmax(energy, axis=1)
        attention = attention / (1e-9 + tf.reduce_sum(attention, axis=-1, keepdims=True)) #[b,n*r,n*r]
        feat_sa = tf.matmul(attention, feat_v)  #[b,n*r,128]
        feat_sa = tf_util.conv1d(feat_sa, 128, 1,
                           padding='VALID', stride=1,
                           bn=True, is_training=is_training, weight_decay=weight_decay,
                           scope='sa_conv1', bn_decay=bn_decay, is_dist=is_dist)
        
        # refinement by self attention
        net = tf_util.conv1d(feat_sa, 64, 1,
                           padding='VALID', stride=1,
                           bn=False, is_training=is_training, weight_decay=weight_decay,
                           scope='sa_conv2', bn_decay=bn_decay, is_dist=is_dist)

        sa_offset = tf_util.conv1d(net, 3, 1,
                           padding='VALID', stride=1,
                           bn=False, is_training=is_training, weight_decay=weight_decay,
                           scope='sa_conv3', bn_decay=bn_decay, is_dist=is_dist, activation_fn=None) # [b,n*r,3]


        S_point2 = S_point1 + sa_offset


    return S_point2, S_point1