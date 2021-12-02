import tensorflow as tf
import os
import numpy as np
import re
from glob import glob

from tf_ops.grouping.tf_grouping import knn_point

is_2D = False


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    batch_size, num_point, num_channels = batch_data.get_shape().as_list()
    angles = tf.clip_by_value(tf.random_normal((batch_size, 3))*angle_sigma, -angle_clip, angle_clip)

    cos_x, cos_y, cos_z = tf.split(tf.cos(angles), 3, axis=-1)  # 3*[B, 1]
    sin_x, sin_y, sin_z = tf.split(tf.sin(angles), 3, axis=-1)  # 3*[B, 1]
    one  = tf.ones_like(cos_x,  dtype=tf.float32)
    zero = tf.zeros_like(cos_x, dtype=tf.float32)
    # [B, 3, 3]
    Rx = tf.stack(
        [tf.concat([one,   zero,   zero], axis=1),
         tf.concat([zero,  cos_x, sin_x], axis=1),
         tf.concat([zero, -sin_x, cos_x], axis=1)], axis=1)

    Ry = tf.stack(
        [tf.concat([cos_y, zero, -sin_y], axis=1),
         tf.concat([zero,  one,    zero], axis=1),
         tf.concat([sin_y, zero,  cos_y], axis=1)], axis=1)

    Rz = tf.stack(
        [tf.concat([cos_z,  sin_z, zero], axis=1),
         tf.concat([-sin_z, cos_z, zero], axis=1),
         tf.concat([zero,   zero,   one], axis=1)], axis=1)

    if is_2D:
        rotation_matrix = Rz
    else:
        rotation_matrix = tf.matmul(Rz, tf.matmul(Ry, Rx))

    if num_channels > 3:
        batch_data = tf.concat(
            [tf.matmul(batch_data[:, :, :3], rotation_matrix),
             tf.matmul(batch_data[:, :, 3:], rotation_matrix),
             batch_data[:, :, 6:]], axis=-1)
    else:
        batch_data = tf.matmul(batch_data, rotation_matrix)

    return batch_data


def random_scale_point_cloud_and_gt(batch_data, batch_gt=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.get_shape().as_list()
    scales = tf.random_uniform((B, 1, 1), minval=scale_low, maxval=scale_high, dtype=tf.float32)

    batch_data = tf.concat([batch_data[:, :, :3] * scales, batch_data[:, :, 3:]], axis=-1)

    if batch_gt is not None:
        batch_gt = tf.concat([batch_gt[:, :, :3] * scales, batch_gt[:, :, 3:]], axis=-1)

    return batch_data, batch_gt, tf.squeeze(scales)


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        assert(clip > 0)
        jittered_data = tf.clip_by_value(sigma * tf.random_normal(tf.shape(batch_data)), -1 * clip, clip)
        if is_2D:
            chn = 2
        else:
            chn = 3
        jittered_data = tf.concat([batch_data[:, :, :chn] + jittered_data[:, :, :chn], batch_data[:, :, chn:]], axis=-1)
        return jittered_data


def rotate_point_cloud_and_gt(batch_data, batch_gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    batch_size, num_point, num_channels = batch_data.get_shape().as_list()
    angles = tf.random_uniform((batch_size, 3), dtype=tf.float32) * 2 * np.pi
    cos_x, cos_y, cos_z = tf.split(tf.cos(angles), 3, axis=-1)  # 3*[B, 1]
    sin_x, sin_y, sin_z = tf.split(tf.sin(angles), 3, axis=-1)  # 3*[B, 1]
    one  = tf.ones_like(cos_x,  dtype=tf.float32)
    zero = tf.zeros_like(cos_x, dtype=tf.float32)
    # [B, 3, 3]
    Rx = tf.stack(
        [tf.concat([one,   zero,   zero], axis=1),
         tf.concat([zero,  cos_x, sin_x], axis=1),
         tf.concat([zero, -sin_x, cos_x], axis=1)], axis=1)

    Ry = tf.stack(
        [tf.concat([cos_y, zero, -sin_y], axis=1),
         tf.concat([zero,  one,    zero], axis=1),
         tf.concat([sin_y, zero,  cos_y], axis=1)], axis=1)

    Rz = tf.stack(
        [tf.concat([cos_z,  sin_z, zero], axis=1),
         tf.concat([-sin_z, cos_z, zero], axis=1),
         tf.concat([zero,   zero,   one], axis=1)], axis=1)

    if is_2D:
        rotation_matrix = Rz
    else:
        rotation_matrix = tf.matmul(Rz, tf.matmul(Ry, Rx))

    if num_channels > 3:
        batch_data = tf.concat(
            [tf.matmul(batch_data[:, :, :3], rotation_matrix),
             tf.matmul(batch_data[:, :, 3:], rotation_matrix),
             batch_data[:, :, 6:]], axis=-1)
    else:
        batch_data = tf.matmul(batch_data, rotation_matrix)

    if batch_gt is not None:
        if num_channels > 3:
            batch_gt = tf.concat(
                [tf.matmul(batch_gt[:, :, :3], rotation_matrix),
                 tf.matmul(batch_gt[:, :, 3:], rotation_matrix),
                 batch_gt[:, :, 6:]], axis=-1)
        else:
            batch_gt = tf.matmul(batch_gt, rotation_matrix)

    return batch_data, batch_gt


def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keepdims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance


class Fetcher(object):
    """docstring for Fetcher"""
    def __init__(self, records, batch_size, num_in_point, num_shape_point, up_ratio_list,
            jitter=False, jitter_max=0.05, jitter_sigma=0.03, drop_out=1.0):
        super(Fetcher, self).__init__()
        self.batch_size = batch_size
        self.num_in_point = num_in_point
        self.jitter = jitter
        self.jitter_max = jitter_max
        self.jitter_sigma = jitter_sigma
        self.drop_out = drop_out
        self.up_ratio_list = up_ratio_list[:]
        record_paths = glob(records)
        print(record_paths)
        saved_patch_size = re.match(".*_p(\d+)_.*", os.path.basename(record_paths[0])).groups()[0]
        saved_patch_size = int(saved_patch_size)
        num_points = re.findall("_(\d+)_", os.path.basename(record_paths[0]))
        num_points = list(map(int, num_points))
        num_points.sort()
        num_points = np.asarray(num_points)
        self.num_shape_point = num_points[np.searchsorted(num_points, num_shape_point)]
        print(self.num_shape_point)
        saved_patch_size = self.num_shape_point / num_points[0] * saved_patch_size
        tag = re.match("^([A-Za-z]+)_\d+", os.path.basename(record_paths[0])).groups()[0]
        self.up_ratio_list.insert(0,1)
        self.features_names = [tag+"_%d" % (self.num_shape_point *up_ratio)
            for up_ratio in self.up_ratio_list]
        self.saved_patch_size = [int(saved_patch_size * up_ratio)
            for up_ratio in self.up_ratio_list]

        print(self.features_names)
        print(self.saved_patch_size)
        self.read_features = {
            name: tf.FixedLenFeature([size, 6], dtype=tf.float32) for name, size in
            zip(self.features_names, self.saved_patch_size)}
        self.ratio_placeholder = tf.placeholder(tf.int32, [])
        label_shapes = tf.constant(self.saved_patch_size[1:], dtype=tf.int32)
        self.offsets = tf.stack([tf.cumsum(label_shapes, exclusive=True),  # 0, n1, n2+n1, ...
                                 tf.cumsum(label_shapes, exclusive=False)], axis=1)

        self.num_batches = 300
        self.dataset = tf.data.TFRecordDataset(record_paths)
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(50))
        self.dataset = self.dataset.map(self.decode, num_parallel_calls=16)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        self.dataset = self.dataset.map(self.get_gt_for_current_ratio)

        self.dataset = self.dataset.map(self.augment_data, num_parallel_calls=16)
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

    def fetch(self, sess):
        return sess.run(self.next_element)

    def initialize(self, sess):
        sess.run(self.iterator.initializer)

    def decode(self, serialized_example):
        read_data = tf.parse_single_example(serialized=serialized_example, features=self.read_features)
        input_pc = read_data[self.features_names[0]]
        label_pc = tf.concat([read_data[k] for k in self.features_names[1:]], axis=0)
        return {"input": input_pc, "label": label_pc}

    def get_gt_for_current_ratio(self, decoded):
        """
        input: B, P, 3
        label: B, P_total, 3
        """
        input_pc, label_pc = decoded["input"], decoded["label"]
        tf_up_ratio_list = tf.cast(self.up_ratio_list[1:], tf.float32)
        probs = tf_up_ratio_list / tf.reduce_sum(tf_up_ratio_list)
        dist = tf.distributions.Categorical(probs=probs, name="ratio_sample_weight")
        pick_ratio_idx = dist.sample([])
        ratio = tf_up_ratio_list[pick_ratio_idx]

        begin = self.offsets[pick_ratio_idx, 0]
        end = self.offsets[pick_ratio_idx, 1]
        radius = tf.constant(np.ones((self.batch_size)), dtype=tf.float32)
        return {"input": input_pc, "label": label_pc[:, begin:end, :], "ratio": ratio, "radius": radius}

    def shape_to_patch(self, label_input_ratio):
        """
        input: [batchsize, P, 3]
        """
        input_pc, label_pc = label_input_ratio['input'], label_input_ratio['label']
        ratio = label_input_ratio["ratio"]
        # B, 1, 1
        rnd_pts = tf.random_uniform((self.batch_size, 1, 1), dtype=tf.int32, maxval=self.saved_patch_size[0])
        batch_indices = tf.reshape(tf.range(self.batch_size), (-1, 1, 1))
        indices = tf.concat([batch_indices, rnd_pts], axis=-1)
        rnd_pts = tf.gather_nd(label_pc, indices)  # [batch_size, 1, 3]
        _, knn_index = knn_point(self.num_in_point*ratio, label_pc, rnd_pts)  # [batch_size, 1, num_gt_point, 2]
        label_patches = tf.gather_nd(label_pc, knn_index)  # [batch_size, 1, num_gt_point, 3]
        _, knn_index = knn_point(self.num_in_point, input_pc, rnd_pts)
        input_patches = tf.gather_nd(input_pc, knn_index)  # [batch_size, 1, num_gt_point/up_ratio, 3]
        label_patches = tf.squeeze(label_patches, axis=1)  # [batch_size, num_gt_point, 3]
        input_patches = tf.squeeze(input_patches, axis=1)

        label_patches, centroid, furthest_distance = normalize_point_cloud(label_patches)
        input_patches = (input_patches - centroid)/furthest_distance
        radius = tf.constant(np.ones((self.batch_size)), dtype=tf.float32)
        return {"label": label_patches, "input": input_patches, "radius": radius, "ratio": ratio}

    def augment_data(self, label_input_radius_ratio):
        input_patches, label_patches, radius = label_input_radius_ratio['input'], label_input_radius_ratio['label'], label_input_radius_ratio['radius']
        ratio = label_input_radius_ratio["ratio"]

        input_patches, label_patches = rotate_point_cloud_and_gt(input_patches, label_patches)
        input_patches, label_patches, scales = random_scale_point_cloud_and_gt(input_patches, label_patches,
            scale_low=0.8, scale_high=1.2)
        radius = radius*scales
        if self.drop_out < 1:
            # randomly discard input
            num_point = input_patches.shape[1].value
            point_idx = tf.random_shuffle(tf.range(num_point))[:int(num_point*self.drop_out)]
            input_patches = tf.gather(input_patches, point_idx, axis=1)
        if self.jitter:
            input_patches = jitter_perturbation_point_cloud(input_patches, sigma=self.jitter_sigma, clip=self.jitter_max)
        return input_patches, label_patches, radius, ratio


if __name__ == '__main__':
    batch_size = 2
    num_point = 256
    num_shape_point = 5000

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    save_path = 'check_db_output'
    TRAIN_RECORD = '/home/qianyue/data/SketchFab/tfrecord_normal_multi/*.tfrecord'

    up_ratio_list = [4,8,12,16]
    fetchworker = Fetcher(TRAIN_RECORD, batch_size = batch_size, up_ratio_list=up_ratio_list, num_in_point=num_point, num_shape_point=num_shape_point, jitter=False, drop_out=1.0)

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        fetchworker.initialize(sess)
        print(fetchworker.num_batches)

        batch_input_data, batch_data_gt, radius, ratio= fetchworker.fetch(sess)  
        print(ratio)
        print(batch_input_data.shape)
        print(batch_data_gt.shape)

        for i in range(batch_size):
            input = batch_input_data[i,:,:]
            gt = batch_data_gt[i,:,:]

