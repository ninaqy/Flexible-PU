import os
import tensorflow as tf

from tf_ops.CD import tf_nndistance
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point,knn_point_2
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
import numpy as np
import math

def get_uniform_loss(pcd, percentages=[0.004,0.006,0.008,0.010,0.012], radius=1.0):
#     B, N, C =  pcd.get_shape().as_list()
    N= 256*4
    npoint = int(N*0.05)
    loss=[]
    for p in percentages:
        nsample = int(N*p)
        r = math.sqrt(p*radius)
        disk_area = math.pi *(radius ** 2) * p/tf.cast(nsample, tf.float32)
        #print(npoint,nsample)
        new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
        idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)#(batch_size, npoint, nsample)

        #expect_len =  tf.sqrt(2*disk_area/1.732)#using hexagon
        expect_len = tf.sqrt(disk_area)  # using square

        grouped_pcd = group_point(pcd, idx)
        grouped_pcd = tf.concat(tf.unstack(grouped_pcd, axis=1), axis=0)

        var, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]
        uniform_dis = tf.sqrt(tf.abs(uniform_dis+1e-8))
        uniform_dis = tf.reduce_mean(uniform_dis,axis=[-1])
        uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
        uniform_dis = tf.reshape(uniform_dis, [-1])

        mean, variance = tf.nn.moments(uniform_dis, axes=0)
        mean = mean*math.pow(p*100,2)
        #nothing 4
        loss.append(mean)
    return tf.add_n(loss)/len(percentages)
    
def get_repulsion_loss4(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss


def get_cd_loss(pred, gt, radius, ratio=0.5):
    """ pred: BxNxC,
        label: BxN, """
    dists_forward,idx1, dists_backward,idx2 = tf_nndistance.nn_distance(gt, pred)
    #dists_forward is for each element in gt, the cloest distance to this element
    CD_dist = ratio*dists_forward + (1-ratio)*dists_backward
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist/radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss, idx1, idx2


def batch_gather(param, idx):
    """
    param: [B, N, C]
    idx: [B, K]
    output: [B, K, C]
    """
    B = param.get_shape()[0].value
    #K = idx.get_shape()[1].value
    K = tf.shape(idx)[1]
    c = tf.convert_to_tensor(np.array(range(B)), dtype=tf.int32)
    c = tf.expand_dims(c,axis=-1)
    c = tf.tile(c,[1,K])
    c = tf.expand_dims(c,axis=-1)
    #c = tf.cast(c, tf.int32)
    idx = tf.expand_dims(idx, axis=-1)
    new = tf.concat([c,idx], axis=-1)

    select = tf.gather_nd(param, new)   
    return select

def get_p2f_loss(pred, gt, radius):
    '''
    pred: [b,n,3]
    gt:  [b,n,6] with normal
    '''
    gt_pc = gt[:,:,0:3]
    dists_forward,idx1, dists_backward,idx2 = tf_nndistance.nn_distance(gt_pc, pred)
    gt_select = batch_gather(gt, idx2)

    dist_select = gt_select[:,:,0:3] - pred
    normal_select = gt_select[:,:,3:6]

    dist1 = tf.multiply(dist_select, normal_select) # check [b,n,3]
    dist2 = tf.reduce_sum(dist1, axis=-1) # check[b,n]
    dist2 = tf.square(dist2) # check[b,n]
    p2f1 = tf.reduce_mean(dist2, axis=1)
    p2f2 = p2f1/radius
    p2f3 = tf.reduce_mean(p2f2)

    return p2f3, None
