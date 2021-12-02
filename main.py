from __future__ import print_function

import argparse
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob
import socket
import matplotlib as mpl
mpl.use('TkAgg') 
import matplotlib.pyplot as plt

import utils.data_provider as data_provider
import utils.model_loss as model_loss
import utils.pc_util as pc_util 
from tf_ops.sampling.tf_sampling import farthest_point_sample

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train', help='train or test [default: train]')
parser.add_argument('--log_dir', default='debug', help='Log dir [default: log]')
parser.add_argument('--pretrain', default=False, action='store_true')
parser.add_argument('--r_train_list', default='4,16', type=str, help='train or test [default: train]')
parser.add_argument('--r_test_list', default='4', type=str, help='train or test [default: train]')
parser.add_argument('--gpu',  default='0',  help='which gpu to use')

parser.add_argument('--train_data',  default='data/train/*.tfrecord',  help='which gpu to use')
parser.add_argument('--test_data',  default='data/test_2048',  help='which gpu to use')

parser.add_argument('--model', default='model_mafu')
parser.add_argument('--loss_uni', type=float, default=1)
parser.add_argument('--loss_mid', type=float, default=30)
parser.add_argument('--loss_p2f', type=float, default=100)

parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 500]')
parser.add_argument('--jitter_sigma', type=float, default=0.01)
parser.add_argument('--jitter_max', type=float, default=0.03)
parser.add_argument('--num_point', type=int, default=256,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--num_shape_point', type=int, default=2048,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--patch_num_ratio', type=int, default=3,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--k', type=int, default=32)

FLAGS = parser.parse_args()

FLAGS.r_train_list = [int(item) for item in FLAGS.r_train_list.split(',')]
FLAGS.r_test_list = [int(item) for item in FLAGS.r_test_list.split(',')]


print(FLAGS)
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
exec('import models.%s as MODEL' % FLAGS.model)

def log_string(out_str):
    print(out_str)
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()

def main(args):
    is_training = True
    bn_decay = 0.95
    step = tf.Variable(0,trainable=False)
    learning_rate = args.learning_rate
    
    tf.summary.scalar('bn_decay', bn_decay)
    tf.summary.scalar('learning_rate', learning_rate)

    # placeholder
    pointclouds_pl, pointclouds_gt, pointclouds_radius, up_ratio_pl, is_training_pl = MODEL.placeholder(args.batch_size, args.num_point)

    # feature extraction and expansion
    pred2, pred1 = MODEL.get_model(pointclouds_pl, is_training_pl, up_ratio=up_ratio_pl, max_ratio=args.r_train_list[-1], k=30, topk=args.k, bradius=pointclouds_radius ,scope='generator')

    # model loss
    gen_loss_cd2, _, _ = model_loss.get_cd_loss(pred2, pointclouds_gt[:,:,0:3], pointclouds_radius, ratio=0.5)
    gen_loss_cd1, _, _ = model_loss.get_cd_loss(pred1, pointclouds_gt[:,:,0:3], pointclouds_radius, ratio=0.5)

    gen_loss_uni = model_loss.get_uniform_loss(pred2)
    gen_loss_p2f, _ = model_loss.get_p2f_loss(pred2[:,:,0:3], pointclouds_gt, pointclouds_radius)
    
    pre_gen_loss = 100 * gen_loss_cd2 + args.loss_mid*gen_loss_cd1  + args.loss_p2f*gen_loss_p2f + args.loss_uni* gen_loss_uni + tf.losses.get_regularization_loss()
   

    # create pre-generator ops
    gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

    with tf.control_dependencies(gen_update_ops):
        pre_gen_train = tf.train.AdamOptimizer(learning_rate,beta1=0.9).minimize(pre_gen_loss,var_list=gen_tvars,
                                                                        global_step=step)
    # merge summary and add pointclouds summary
    tf.summary.scalar('loss/gen_cd2', gen_loss_cd2)
    tf.summary.scalar('loss/gen_cd1', gen_loss_cd1)
    tf.summary.scalar('loss/gen_p2f', gen_loss_p2f)
    tf.summary.scalar('loss/gen_uni', gen_loss_uni)
    tf.summary.scalar('loss/regularation', tf.losses.get_regularization_loss())
    tf.summary.scalar('loss/pre_gen_total', pre_gen_loss)
    pretrain_merged = tf.summary.merge_all()

    # create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'train'), sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})
        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_gt': pointclouds_gt,
               'pointclouds_radius': pointclouds_radius,
               'pretrain_merged':pretrain_merged,
               'gen_loss_cd2': gen_loss_cd2,
               'gen_loss_cd1': gen_loss_cd1,
               'gen_loss_p2f': gen_loss_p2f,
               'gen_loss_uni': gen_loss_uni,
               'pre_gen_loss': pre_gen_loss,
               'pre_gen_train':pre_gen_train,
               'pred2': pred2,
               'pred1': pred1,
               'step': step,
               'is_training_pl': is_training_pl,
               'up_ratio_pl': up_ratio_pl
               }
   

        saver = tf.train.Saver(max_to_keep=2)
        global LOG_FOUT
        if args.pretrain:
            restore_epoch, checkpoint_path = pc_util.pre_load_checkpoint(args.log_dir, latest_filename = 'checkpoint')
            LOG_FOUT = open(os.path.join(args.log_dir, 'log_max_%d.txt'%args.r_train_list[-1]), 'a')
            saver.restore(sess,checkpoint_path)
            print('Loaded: ', checkpoint_path)
            print('Epoch trained %d' % restore_epoch)
        else: 
            restore_epoch = 0 
            LOG_FOUT = open(os.path.join(args.log_dir, 'log_max_%d.txt'%args.r_train_list[-1]), 'w')
            LOG_FOUT.write(str(socket.gethostname()) + '\n')
            LOG_FOUT.write(str(FLAGS) + '\n')

        fetchworker = data_provider.Fetcher(args.train_data, batch_size = args.batch_size, up_ratio_list=args.r_train_list, num_in_point=args.num_point, num_shape_point=args.num_shape_point, jitter=True, drop_out=1.0, jitter_max=args.jitter_max, jitter_sigma=args.jitter_sigma)
    
        if args.phase == 'train':
            for epoch in tqdm(range(args.max_epoch+1),ncols=55):
                train_one_epoch(sess, ops, args, fetchworker, train_writer)
                if epoch % 50 ==0:
                    saver.save(sess, os.path.join(args.log_dir, "model"), global_step=epoch)
        
        elif args.phase == 'test':
            for up_ratio in args.r_test_list:
                print(up_ratio)
                path_input = FLAGS.test_data
                path_output = os.path.join(args.log_dir, 'test_x%d/' %up_ratio) 
                eval_whole_model(sess, ops, args, up_ratio, path_input, path_output)
                
def train_one_epoch(sess, ops, args, fetchworker, train_writer):
    epsilon = 1e-7
    is_training = True
    loss_total_sum, loss_sum_cd2, loss_sum_cd1, loss_sum_p2f, loss_sum_uni = {},{},{},{}, {}
    for ratio in args.r_train_list:
        loss_total_sum[ratio] = []
        loss_sum_cd2[ratio] = []
        loss_sum_cd1[ratio] = []
        loss_sum_p2f[ratio] = []
        loss_sum_uni[ratio] = []

    fetchworker.initialize(sess)

    for batch_idx in range(fetchworker.num_batches):
        batch_input_data, batch_data_gt, radius, up_ratio = fetchworker.fetch(sess)  

        batch_gt_normal = batch_data_gt[:,:,3:6]
        gt_normal_l2 = np.linalg.norm(batch_gt_normal, axis=-1, keepdims=True)
        gt_normal_l2 = np.tile(gt_normal_l2, [1,3])
        batch_gt_normal = np.divide(batch_gt_normal,gt_normal_l2+epsilon)
        batch_data_gt[:,:,3:6] = batch_gt_normal

        feed_dict = {ops['pointclouds_pl']: batch_input_data[:,:,0:3],
                    ops['pointclouds_gt']: batch_data_gt[:,:,0:6],
                     ops['is_training_pl']: is_training,
                     ops['pointclouds_radius']: radius,
                     ops['up_ratio_pl']: int(up_ratio)
                    }
        summary, step, _, pre_gen_loss,gen_loss_cd2, gen_loss_cd1, gen_loss_p2f, gen_loss_uni = sess.run( [ops['pretrain_merged'],ops['step'],ops['pre_gen_train'],ops['pre_gen_loss'], ops['gen_loss_cd2'],  ops['gen_loss_cd1'], ops['gen_loss_p2f'], ops['gen_loss_uni']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_total_sum[up_ratio].append(pre_gen_loss)
        loss_sum_cd2[up_ratio].append(gen_loss_cd2)
        loss_sum_cd1[up_ratio].append(gen_loss_cd1)
        loss_sum_p2f[up_ratio].append(gen_loss_p2f)
        loss_sum_uni[up_ratio].append(gen_loss_uni)


    for ratio in loss_total_sum:
        loss_total_sum_array = np.asarray(loss_total_sum[ratio])
        loss_sum_cd2_array = np.asarray(loss_sum_cd2[ratio])
        loss_sum_cd1_array = np.asarray(loss_sum_cd1[ratio])
        loss_sum_p2f_array = np.asarray(loss_sum_p2f[ratio])
        loss_sum_uni_array = np.asarray(loss_sum_uni[ratio])

        log_string('step: %d, ratio: %d, total: %f, cd: %f, cd mid: %f, p2f: %f,  uni: %f\n' % (step, ratio, round(loss_total_sum_array.mean(),7), round(loss_sum_cd2_array.mean(),7), round(loss_sum_cd1_array.mean(),7) ,round(loss_sum_p2f_array.mean(),7), round(loss_sum_uni_array.mean(),7)))     


def patch_prediction_eval(pc_point, sess, ops, args, ratio):
    is_training = False
    
    # normalize the point clouds
    patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(pc_point)
    patch_point = np.expand_dims(patch_point,axis=0)
    
    batch_patch_point = np.tile(patch_point, [args.batch_size, 1, 1])
    batch_pred2, batch_pred1 = sess.run([ops['pred2'], ops['pred1'] ],
        feed_dict={ops['pointclouds_pl']: batch_patch_point, 
            ops['is_training_pl']: is_training,
            ops['pointclouds_radius']: np.ones([args.batch_size], dtype='f'),
            ops['up_ratio_pl']: ratio})
    

    pred2 = np.expand_dims(batch_pred2[0], axis=0)
    pred2_pc = pred2[:,:,0:3]
    pred2_pc = np.squeeze(centroid + pred2_pc * furthest_distance, axis=0)

    pred1 = np.expand_dims(batch_pred1[0], axis=0)
    pred1_pc = pred1[:,:,0:3]
    pred1_pc = np.squeeze(centroid + pred1_pc * furthest_distance, axis=0)
    
    return pred2_pc, pred1_pc


def pc_prediction_eval(pc, sess, ops, args, patch_num_ratio, ratio):
    num_shape_point = pc.shape[0]
    ## FPS sampling
    points = tf.convert_to_tensor(np.expand_dims(pc,axis=0),dtype=tf.float32)
    seed1_num = int(num_shape_point/ args.num_point * args.patch_num_ratio)
    seed = farthest_point_sample(seed1_num, points).eval()[0]
    seed_list = seed[:seed1_num]

    print("number of patches: %d" % len(seed_list))
    input_list = []
    up_point2_list=[]
    up_point1_list=[]

    patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, args.num_point)
    for i in tqdm(range(len(patches))):
        point = patches[i]
        up_point2, up_point1 = patch_prediction_eval(point, sess, ops, args, ratio)

        input_list.append(point)
        up_point2_list.append(up_point2)
        up_point1_list.append(up_point1)

    return input_list, up_point2_list, up_point1_list


def eval_whole_model(sess, ops, args, up_ratio, path_input, path_output):
    # get necessary parameters
    num_point = args.num_point
    num_shape_point = args.num_shape_point
    patch_num_ratio = 3
    
    if not os.path.exists(path_output): 
        os.makedirs(path_output)

    # obtain xyz file from path_input
    pcs_input = glob(path_input + "/*.xyz")
    num_pcs = len(pcs_input)
    print('total obj %d' %num_pcs)
    
    for i,path_input_xyz in enumerate(pcs_input):
        pc_input = np.loadtxt(path_input_xyz)
        name_obj = path_input_xyz.split('/')[-1] # with .xyz
        pc_input = pc_input[:, 0:3]
        pc_input_normed, centroid, scale = pc_util.normalize_point_cloud(pc_input)
        
        # obtain patch prediction
        input_list, pred2_list, pred1_list = pc_prediction_eval(pc_input_normed, sess, ops, args, patch_num_ratio=patch_num_ratio, ratio=up_ratio)       

        # formulate patch prediction to full model by fps
        pred2_normed = np.concatenate(pred2_list, axis=0)
        idx = farthest_point_sample(num_shape_point*up_ratio, pred2_normed[np.newaxis,...]).eval()[0]
        pred2_normed = pred2_normed[idx,0:3]  
        pred2 = (pred2_normed*scale) + centroid

        # save xyz
        save_path = os.path.join(path_output, 'input_'+name_obj)
        np.savetxt(save_path, pc_input)
        save_path = os.path.join(path_output, 'pred_'+name_obj)
        np.savetxt(save_path, pred2)
              
if __name__ == "__main__":
    main(FLAGS)       
    LOG_FOUT.close()

