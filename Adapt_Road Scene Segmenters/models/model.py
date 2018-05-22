from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import ceil
from zero_gradient import zero_gradient

import os
import logging
import sys
import numpy as np
import tensorflow as tf
import pdb


# in BGR order
DataSet_Mean = {'Taipei':[104.03, 104.93, 103.30],
                'Tokyo':[120.04, 121.09, 119.94],
                'Denmark':[126.77, 130.34, 127.37],
                'Roma':[113.44, 115.97, 114.38],
                'Rio':[115.81, 118.83, 116.11],
                'Cityscapes':[72.39, 82.91, 73.16]}

class FCN8VGG:
    def __init__(self, vgg16_npy_path=None):
        
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "../cscap_dlvgg16.npy")
            vgg16_npy_path = path
            logging.info("Load npy file from '%s'.", vgg16_npy_path)
        
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.wd = 5e-4
        print("npy file loaded")
    
    #def build(self, rgb, task_labels, domain_label, batch_size, train=False, num_classes=19, city='Taipei', 
    def build(self, batch_size, train=False, num_classes=19, city='Taipei', 
              random_init_fc8=False, random_init_adnn=False, debug=False):
        
        src_mean = DataSet_Mean['Cityscapes']
        tgt_mean = DataSet_Mean[city]

        self.rgb = tf.placeholder(tf.int32, shape = [batch_size, 256, 512, 3], name = 'rgb')
        self.task_labels = tf.placeholder(tf.float32, shape = [batch_size, 256, 512, 1], name = 'task_labels') 
        
        self.domain_labels = tf.placeholder(tf.float32, shape = [batch_size, 32, 64, 1], name = 'domain_labels')
        domain_label = tf.cast(tf.squeeze(self.domain_labels, [3]), tf.int32)
        # Convert RGB to BGR
        with tf.name_scope('Processing'):

            if train:
                src_planes = tf.split(self.rgb[:int(batch_size/2), :, :, :], 3, 3)
                r_src, g_src, b_src = [tf.cast(plane, tf.float32) for plane in src_planes]
                
                tgt_planes = tf.split(self.rgb[int(batch_size/2):, :, :, :], 3, 3)
                r_tgt, g_tgt, b_tgt = [tf.cast(plane, tf.float32) for plane in tgt_planes]
                bgr_src = tf.concat([b_src - src_mean[0],
                                     g_src - src_mean[1],
                                     r_src - src_mean[2]], 3)
            
                bgr_tgt = tf.concat([b_tgt - tgt_mean[0],
                                     g_tgt - tgt_mean[1],
                                     r_tgt - tgt_mean[2]], 3)
                self.bgr = tf.concat([bgr_src, bgr_tgt], 0)
            else:
                tgt_planes = tf.split(self.rgb, 3, 3)
                r_tgt, g_tgt, b_tgt = [tf.cast(plane, tf.float32) for plane in tgt_planes]
                
                self.bgr = tf.concat([b_tgt - tgt_mean[0],
                                      g_tgt - tgt_mean[1],
                                      r_tgt - tgt_mean[2]], 3)
        
        with tf.variable_scope('feature_extractor'):
            
            self.conv1_1 = self._conv_layer(self.bgr, "conv1_1")
            self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)
            
            self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)
            
            self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)
        
            self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
            
            self.conv5_1 = self._dilated_conv_layer(self.conv4_3, "conv5_1", 2)
            self.conv5_2 = self._dilated_conv_layer(self.conv5_1, "conv5_2", 2)
            self.conv5_3 = self._dilated_conv_layer(self.conv5_2, "conv5_3", 2) 
            
            self.fc6 = self._dilated_conv_layer(self.conv5_3, "fc6", 4)
            if train:
                self.fc6 = tf.nn.dropout(self.fc6, 0.5)
            
            self.fc7 = self._fc_layer(self.fc6, "fc7")
            if train:
                self.fc7 = tf.nn.dropout(self.fc7, 0.5)
        
        with tf.variable_scope('label_predictor'):
            
            if random_init_fc8:
                self.final = self._score_layer(self.fc7, "final", num_classes)
            else:
                self.final = self._fc_layer(self.fc7, "final",
                                            num_classes=num_classes,
                                            relu=False)
            self.upsample = self._upscore_layer(self.final,
                                                shape=self.bgr.get_shape(),
                                                num_classes=num_classes,
                                                debug=debug, name='upsample',
                                                ksize=16, stride=8)
            self.pred_prob = tf.nn.softmax(self.upsample, name='pred_prob')
            self.pred_up = tf.argmax(self.upsample, dimension=3) # for inference
        
        if train:
            
            #####################################
            ##############Task loss##############
            #####################################
        
            # 255 stands for ignored label
            task_labels = tf.cast(tf.gather(self.task_labels, tf.range(int(batch_size/2))), tf.int32)
            task_labels = tf.squeeze(task_labels, [3]) 
            mask = tf.constant(255, shape=task_labels.get_shape().as_list())
            mask = tf.cast(tf.not_equal(task_labels, mask), tf.int32)
            valid_pixel_num = tf.cast(tf.reduce_sum(mask), tf.float32)
            
            # calculating the accuracy
            prediction = tf.cast(tf.gather(self.pred_up, tf.range(int(batch_size/2))), tf.int32) 
            self.task_accur = tf.div(tf.reduce_sum(tf.cast(tf.equal(prediction,task_labels),tf.float32)), valid_pixel_num)
            
            # calculating the loss
            task_labels = tf.multiply(task_labels, mask) # remove those pixels labeled as 255
            epsilon = tf.constant(value=1e-4)
            logits = tf.gather(self.upsample, tf.range(int(batch_size/2))) + epsilon
                        
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = task_labels, logits = logits, name = 'cross_entropy_per_example')
            cross_entropy = tf.multiply(cross_entropy, tf.cast(mask, dtype=tf.float32)) # igonre the loss of pixels labeled by 255
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean') # average over the batch
            tf.add_to_collection('losses', cross_entropy_mean)
            self.task_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
        ######################################
        ###########adversarial net############
        ######################################

        # GA stands for global alignment    
        # CA stands for class-specific alignment
        
        if random_init_adnn and train:
            
            with tf.variable_scope('global_alignment'):
                    
                # discriminator for global alignment
                self.GA_adnn1 = self._score_layer(self.fc7, "GA_adnn1")
                self.GA_adnn1 = tf.nn.relu(self.GA_adnn1)
                self.GA_adnn1 = tf.nn.dropout(self.GA_adnn1, 0.5)
                
                self.GA_adnn2 = self._score_layer(self.GA_adnn1, "GA_adnn2")
                self.GA_adnn2 = tf.nn.relu(self.GA_adnn2)
                self.GA_adnn2 = tf.nn.dropout(self.GA_adnn2, 0.5)

                self.GA_adnn3 = self._score_layer(self.GA_adnn2, "GA_adnn3")
                GA_domain_pred = tf.cast(tf.argmax(self.GA_adnn3, dimension=3), tf.int32)
                self.GA_domain_accur = tf.reduce_mean(tf.cast(tf.equal(GA_domain_pred, domain_label), tf.float32))
                GA_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                     labels = domain_label,
                                     logits = self.GA_adnn3)
                GA_entropy_inv = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                     labels = 1-domain_label,
                                     logits = self.GA_adnn3)
                
                self.GA_domain_loss = tf.reduce_mean(GA_entropy, name='GA_loss')
                self.GA_domain_loss_inv = tf.reduce_mean(GA_entropy_inv, name='GA_loss_inv')

            with tf.variable_scope('class-specific_alignment'):
                
                # discriminator for class-specific alignment 
                task_labels = zero_gradient(task_labels)
                psuedo_labels = zero_gradient(tf.gather(self.pred_prob, tf.range(int(batch_size/2), batch_size)))

                self.CA_adnn1s = list()
                self.CA_adnn2s = list()
                self.CA_adnn3s = list()
                
                self.CA_domain_accur_by_cls = list()

                self.CA_domain_loss = 0
                self.CA_domain_loss_inv = 0
                self.CA_domain_accur = 0

                for i in range(num_classes):
                    
                    CA_adnn1 = self._score_layer(self.fc7, "CA%d_adnn1" % i)
                    CA_adnn1 = tf.nn.relu(CA_adnn1)
                    CA_adnn1 = tf.nn.dropout(CA_adnn1, 0.5)
                    self.CA_adnn1s.append(CA_adnn1)
                    CA_adnn2 = self._score_layer(self.CA_adnn1s[i], "CA%d_adnn2" % i)
                    CA_adnn2 = tf.nn.relu(CA_adnn2)
                    CA_adnn2 = tf.nn.dropout(CA_adnn2, 0.5)
                    self.CA_adnn2s.append(CA_adnn2)
                    self.CA_adnn3s.append(self._score_layer(self.CA_adnn2s[i], "CA%d_adnn3" % i)) # with shape (batch, fc7_h, fc7_w, 2)
                    
                    CA_domain_pred = tf.cast(tf.argmax(self.CA_adnn3s[i], dimension=3), tf.int32)
                    
                    #self.CA_domain_accur += tf.reduce_mean(tf.cast(tf.equal(CA_domain_pred, domain_label), tf.float32))
                    #self.CA_domain_accur_by_cls.append(tf.reduce_mean(tf.cast(tf.equal(CA_domain_pred, domain_label), tf.float32)))
                    
                    # the shape of CA_entropy & CA_entropy_inv is (batch, fc7_h, fc7_w)
                    CA_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels = domain_label,
                                    logits = self.CA_adnn3s[i])
                    
                    CA_entropy_inv = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        labels = 1-domain_label,
                                        logits = self.CA_adnn3s[i])
                    
                    ##############################################
                    # iterate through each grid in fc7 feature map
                    # calulate the weighting of each grid
                    ##############################################

                    # the shape of mask is (batch/2, fc7_h * 8, fc7_w * 8)
                    mask = tf.constant(i, shape=task_labels.get_shape().as_list())
                    mask = tf.cast(tf.equal(task_labels, mask), tf.float32)
                    mask = tf.expand_dims(mask, 3)

                    # the shape of cw_psuedo_labels is (batch/2, fc7_h * 8, fc7_w * 8)
                    # extract the corresponding psuedo label of each class
                    cw_psuedo_labels = tf.transpose(tf.gather(tf.transpose(psuedo_labels), i))
                    cw_psuedo_labels = tf.expand_dims(cw_psuedo_labels, 3)

                    src_weighting = tf.nn.avg_pool(mask, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
                    tgt_weighting = tf.nn.avg_pool(cw_psuedo_labels, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
                    
                    src_weighting = tf.squeeze(src_weighting, [3])
                    tgt_weighting = tf.squeeze(tgt_weighting, [3])
                    
                    weighting = tf.concat([src_weighting, tgt_weighting], 0) 
                    
                    weighting_sum_by_img = tf.reduce_sum(weighting, [1,2])
                    small_constant = tf.ones(weighting_sum_by_img.get_shape()) * 5e-6
                    weighting_sum_by_img = tf.add(weighting_sum_by_img, small_constant)

                    # calculate the weighted domain accur
                    weighted_CA_domain_accur_by_img = tf.reduce_sum(tf.multiply(tf.cast(tf.equal(CA_domain_pred, domain_label), tf.float32), \
                                                                                weighting), [1,2])
                    
                    weighted_CA_domain_accur = tf.reduce_mean(tf.div(weighted_CA_domain_accur_by_img,
                                                                     weighting_sum_by_img))
                    
                    self.CA_domain_accur += weighted_CA_domain_accur
                    self.CA_domain_accur_by_cls.append(weighted_CA_domain_accur)
                    
                    # calculate the weighted entropy
                    weighted_entropy = tf.multiply(CA_entropy, weighting)
                    weighted_entropy_inv = tf.multiply(CA_entropy_inv, weighting)
                    
                    self.CA_domain_loss += tf.reduce_mean(tf.div(tf.reduce_sum(weighted_entropy, [1,2]), weighting_sum_by_img))
                    self.CA_domain_loss_inv += tf.reduce_mean(tf.div(tf.reduce_sum(weighted_entropy_inv, [1,2]), weighting_sum_by_img))
                
                self.CA_domain_accur = self.CA_domain_accur / (num_classes)
        
        t_vars = tf.trainable_variables()
        
        self.f_vars = [var for var in t_vars if 'feature_extractor' in var.name]
        self.y_vars = [var for var in t_vars if 'label_predictor' in var.name]
        
        if train:
            self.ga_vars = [var for var in t_vars if 'global_alignment' in var.name]
            self.ca_vars = [var for var in t_vars if 'class-specific_alignment' in var.name]

    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu
    
    def _dilated_conv_layer(self, bottom, name,rate):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            #conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.atrous_conv2d(bottom, filt, rate, padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu


    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'final': #'score_fr':
                #name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 19],
                                                  num_classes=num_classes)
            else: #fc7
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            self._add_wd_and_summary(filt, self.wd, "fc_wlosses")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _score_layer(self, bottom, name, num_classes=20):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name.split('_')[-1] in ['adnn1', 'adnn2']:
                shape = [1, 1, in_features, 1024]
            elif name.split('_')[-1] == 'adnn3':
                shape = [1, 1, in_features, 2]
            elif name.split('_')[-1] == 'wgan':
                shape = [1, 1, in_features, 1]

            if name == "final": #"score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            elif name == "score_pool3":
                stddev = 0.0001
            elif name.split('_')[-1] in ['adnn1', 'adnn2', 'adnn3', 'wgan']:
                stddev = 0.001
            #elif name.split('_')[-1] in ['adnn2', 'wgan']:
            #    stddev = 0.001
            # Apply convolution
            w_decay = self.wd

            weights = self._variable_with_weight_decay(shape, stddev, w_decay,
                                                       decoder=True)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([shape[3]], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [int(shape[0]), int(shape[1]), int(shape[2]), num_classes]
            
            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter('upsample',f_shape)
            self._add_wd_and_summary(weights, self.wd, "fc_wlosses")
            deconv = tf.nn.conv2d_transpose(bottom, weights, new_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, name, f_shape):
        """
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear
        """
        weights_2 = np.zeros(f_shape)
        w_in = self.data_dict[name][0]
        for i in range(f_shape[2]):
            weights_2[:, :, i, i] = w_in[:,:,i]
        #weights_2 = np.tile(np.expand_dims(weights_2, 2), (1, 1, 19, 1))
        
        weights = weights_2.reshape(f_shape)

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init,
                              shape=weights.shape)
        
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.nn.l2_loss(var)* self.wd
            tf.add_to_collection('losses', weight_decay)
        
        return var

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        #value = self.data_dict[name][0][0][0][0][0]
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        #print('Layer value: %s' % str(value))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.nn.l2_loss(var)* self.wd
            tf.add_to_collection('losses', weight_decay)
        _variable_summaries(var)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'final': #'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        _variable_summaries(var)
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.nn.l2_loss(var)* self.wd
            tf.add_to_collection('losses', weight_decay)
        _variable_summaries(var)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd, decoder=False):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.nn.l2_loss(var)* wd
            if not decoder:
                tf.add_to_collection('losses', weight_decay)
            else:
                tf.add_to_collection('dec_losses', weight_decay)
        _variable_summaries(var)
        return var

    def _add_wd_and_summary(self, var, wd, collection_name="losses"):
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.nn.l2_loss(var)* wd
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        var = tf.get_variable(name='biases', shape=shape,
                              initializer=initializer)
        _variable_summaries(var)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        #pdb.set_trace()
        #if((name!='fc7') and (name!='final')):
        weights = weights.reshape(shape)
        #if num_classes is not None:
        #    weights = self._summary_reshape(weights, shape,
        #                                    num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var
    


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    #tf.histogram_summary(tensor_name + '/activations', x)
    #tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            #tf.scalar_summary(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            #tf.scalar_summary(name + '/sttdev', stddev)
            #tf.scalar_summary(name + '/max', tf.reduce_max(var))
            #tf.scalar_summary(name + '/min', tf.reduce_min(var))
            #tf.histogram_summary(name, var)
