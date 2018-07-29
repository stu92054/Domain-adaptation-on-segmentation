import sys
import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import os
from src.custom_grad import WeakLoss
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weight_path', type=str, required=True)
parser.add_argument('--city', type=str, default='Taipei')
parser.add_argument('--src_data_path', type=str, required=True)
parser.add_argument('--tgt_data_path', type=str, required=True)
parser.add_argument('--method', type=str, default='GA')
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--iter_size', type=int, required=True)
parser.add_argument('--max_step', type=int, required=True)
parser.add_argument('--start_step', type=int, default=0)
parser.add_argument('--save_step', type=int, required=True)
parser.add_argument('--train_dir', type=str, default='./trained_weight/')
parser.add_argument('--input_width', type=int, default=512)
parser.add_argument('--input_height', type=int, default=256)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--restore_path', type=str, required=False)
parser.add_argument('--write_summary', type=bool, default=False)

args = parser.parse_args()
weight_path = args.weight_path
city = args.city
src_data_path = args.src_data_path 
tgt_data_path = args.tgt_data_path
method = args.method
batch_size = args.batch_size
iter_size = args.iter_size
save_step = args.save_step
max_step = args.max_step
start_step = args.start_step
train_dir = args.train_dir
input_width = args.input_width
input_height = args.input_height
restore_path = args.restore_path
train_method_city_dir = train_dir + method+ '/'+ city + '/'

model_path = './models/'
sys.path.insert(0, model_path)

### select the specific module of model and data according to the method ###
if method == 'GA':
    print 'GA only!'
    _CW_Alignment = False

elif method == 'GACA':
    print 'GA + CA!'
    _CW_Alignment = True

from model import FCN8VGG
from data_reader import Reader

if not os.path.isfile(weight_path):
    raise IOError("Error: Pre-trained model doesn't exist!")
if not os.path.isfile(src_data_path) or not os.path.isfile(tgt_data_path):
    raise IOError("Error: Data doesn't exist!")
if not os.path.isdir(train_method_city_dir):
    os.makedirs(train_method_city_dir)
if not os.path.isdir('./logfiles/'):
    os.mkdir('./logfiles/')

assert iter_size > 1, 'iter_size should be larger than 1!' 
assert city in ['Taipei', 'Roma', 'Tokyo', 'Rio', 'Denmark','syn2real'], 'Please check the city name!'
assert method in ['GA', 'GACA'], 'Please check the method name!'

reader = Reader(src_data_path, tgt_data_path, input_width=input_width, input_height=input_height, batch_size=batch_size)
model = FCN8VGG(weight_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# create additional class-specific loss layer
if city in ['Taipei', 'Roma', 'Tokyo', 'Rio', 'Denmark']:
    weak_loss = WeakLoss('Cityscapes') # create object and assign src dataset
else:
    weak_loss = WeakLoss('Synthia') 

def cal_grad_func_impl(x, grad):
    return weak_loss.diff * grad    # grad = 1.0 in lossLayer

def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):
    grad_name = 'PyFuncGrad_' + str(np.random.randint(0, 1e+8))
    tf.RegisterGradient(grad_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def cal_grad_func(op, grad):
    return tf.py_func(cal_grad_func_impl, [model.upsample, grad], grad.dtype)

def step_function(x, start, end):
    if x <=start:
        return 0.0
    elif x > start and x <= end:
        return 1./(end-start) * (x-start)
    else:
        return 1.

def train():
    with tf.Graph().as_default():
        
        with tf.device('/gpu:0'):
            model.build(batch_size=batch_size,img_w=input_width, img_h=input_height,
                        train=True, num_classes=19, city=city, 
                        random_init_fc8=False, random_init_adnn=True, debug=True)
        
        lr = tf.placeholder(tf.float32, shape = [], name = 'lr')
        # for accumulating and reseting the gradients
        # mode = 0 stands for accumulating the gradients
        # mode = 1 stands for applying the gradients
        # mode = 2 stands for reseting the gradients
        mode = tf.placeholder(tf.float32, shape = [], name = 'mode')
         
        task_loss = model.task_loss
        task_accur = model.task_accur

        GA_domain_loss = model.GA_domain_loss
        
        GA_domain_accur = model.GA_domain_accur
        GA_confusion_loss = tf.div(tf.add(model.GA_domain_loss, model.GA_domain_loss_inv), 2.0)
        
        # weakloss
        cw_loss = py_func(weak_loss.calculate,
                        [model.upsample], [tf.float32], name='np_cal',                                              
                        grad_func=cal_grad_func)
        cw_loss = cw_loss[0]

            
        # summary on tensorboard        
        tf.summary.scalar('task_loss', task_loss)
        tf.summary.scalar('task_accur', task_accur)
        tf.summary.scalar('GA_domain_loss', GA_domain_loss)
        tf.summary.scalar('GA_domain_accur', GA_domain_accur)
        
        ##########################
        ## accumulate gradients ##
        ##########################
        
        # f stands for feature extractor 
        # y stands for label predictor
        # d stands for domain discriminator

        f_vars = model.f_vars
        y_vars = model.y_vars
        ga_vars = model.ga_vars
        #ca_vars = model.ca_vars
        
        stored_grads_task = [tf.Variable(tf.zeros(var.get_shape()), trainable = False) for var in (f_vars + y_vars)]
        accum_grads_task = [tf.zeros_like(var) for var in (f_vars + y_vars)]
        applied_grads_task = [tf.zeros_like(var) for var in (f_vars + y_vars)]
        reset_grads_task = [tf.zeros_like(var) for var in (f_vars + y_vars)]
        
        stored_grads_GA = [tf.Variable(tf.zeros(var.get_shape()), trainable = False) for var in (ga_vars)]
        accum_grads_GA = [tf.zeros_like(var) for var in (ga_vars)]
        applied_grads_GA = [tf.zeros_like(var) for var in (ga_vars)]
        reset_grads_GA = [tf.zeros_like(var) for var in (ga_vars)]

        stored_grads_GA_inv = [tf.Variable(tf.zeros(var.get_shape()), trainable = False) for var in f_vars]
        accum_grads_GA_inv = [tf.zeros_like(var) for var in f_vars]
        applied_grads_GA_inv = [tf.zeros_like(var) for var in f_vars]
        reset_grads_GA_inv = [tf.zeros_like(var) for var in f_vars]
        
        
        # if mode == 2, reset the gradients
        for index in range(len(stored_grads_task)):
            reset_grads_task[index] = tf.cond(tf.equal(mode, 2),
                                         lambda: stored_grads_task[index].assign(tf.zeros_like(stored_grads_task[index])),
                                         lambda: tf.ones_like(stored_grads_task[index]))
        
        for index in range(len(stored_grads_GA)):
            reset_grads_GA[index] = tf.cond(tf.equal(mode, 2),
                                         lambda: stored_grads_GA[index].assign(tf.zeros_like(stored_grads_GA[index])),
                                         lambda: tf.ones_like(stored_grads_GA[index]))
        
        for index in range(len(stored_grads_GA_inv)):
            reset_grads_GA_inv[index] = tf.cond(tf.equal(mode, 2),
                                         lambda: stored_grads_GA_inv[index].assign(tf.zeros_like(stored_grads_GA_inv[index])),
                                         lambda: tf.ones_like(stored_grads_GA_inv[index]))
         
        opt_task = tf.train.AdamOptimizer(learning_rate = lr)
        opt_GA = tf.train.AdamOptimizer(learning_rate = lr)
        opt_GA_inv = tf.train.AdamOptimizer(learning_rate = lr)
        opt_CA = tf.train.AdamOptimizer(learning_rate = lr)
        
        grads_task= tf.gradients(task_loss, f_vars + y_vars)
        
        grads_GA= tf.gradients(GA_domain_loss, ga_vars)
        grads_GA_inv = tf.gradients(GA_confusion_loss, f_vars)
        
        grads_CA = tf.gradients(cw_loss, f_vars + y_vars)
        
        # accumulating the calculated gradients
        for index in range(len(stored_grads_task)):
            accum_grads_task[index] = stored_grads_task[index].assign_add(grads_task[index])
            applied_grads_task[index] = tf.scalar_mul(1.0/iter_size, accum_grads_task[index])
        
        for index in range(len(stored_grads_GA)):
            accum_grads_GA[index] = stored_grads_GA[index].assign_add(grads_GA[index])
            applied_grads_GA[index] = tf.scalar_mul(1.0/iter_size, accum_grads_GA[index])
       
        for index in range(len(stored_grads_GA_inv)):
            accum_grads_GA_inv[index] = stored_grads_GA_inv[index].assign_add(grads_GA_inv[index])
            applied_grads_GA_inv[index] = tf.scalar_mul(1.0/iter_size, accum_grads_GA_inv[index])
        

        train_op_task = tf.cond(tf.equal(mode, 1),
                              lambda: opt_task.apply_gradients(zip(applied_grads_task, f_vars + y_vars)),
                              lambda: tf.no_op())
        
        train_op_GA = tf.cond(tf.equal(mode, 1),
                              lambda: opt_GA.apply_gradients(zip(applied_grads_GA, ga_vars)),
                              lambda: tf.no_op())
        
        train_op_GA_inv = tf.cond(tf.equal(mode, 1),
                                  lambda: opt_GA_inv.apply_gradients(zip(applied_grads_GA_inv, f_vars)),
                                  lambda: tf.no_op())
        
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar('learning_rate', lr))
        variable_averages = tf.train.ExponentialMovingAverage(0.9)
        
        f_vars_avg_op = variable_averages.apply(f_vars)
        y_vars_avg_op = variable_averages.apply(y_vars)
        ga_vars_avg_op = variable_averages.apply(ga_vars)
        #ca_vars_avg_op = variable_averages.apply(ca_vars)
        
        summary_op = tf.summary.merge(summaries)
        
        train_op_task = tf.group(train_op_task, f_vars_avg_op)
        train_op_task = tf.group(train_op_task, y_vars_avg_op)
        train_op_GA = tf.group(train_op_GA, ga_vars_avg_op)
        train_op_GA_inv = tf.group(train_op_GA_inv, f_vars_avg_op)
        
        for index in range(len(stored_grads_task)):
            train_op_task = tf.group(train_op_task, reset_grads_task[index])
        
        for index in range(len(stored_grads_GA)):
            train_op_GA = tf.group(train_op_GA, reset_grads_GA[index])
        
        for index in range(len(stored_grads_GA_inv)):
            train_op_GA_inv = tf.group(train_op_GA_inv, reset_grads_GA_inv[index])
        
        train_op_CA = opt_CA.apply_gradients(zip(grads_CA, f_vars + y_vars))


        summary = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op_task = tf.summary.merge([v for v in summary if v.name.startswith("task")])
        summary_op_GA = tf.summary.merge([v for v in summary if v.name.startswith("GA")])
        
        sess = tf.InteractiveSession(config = tf.ConfigProto(allow_soft_placement = True,
                                                  log_device_placement = True))

        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
                                                  log_device_placement = True))
    
        # Build an initialization operation to run below.
        tf.global_variables_initializer().run(session = sess)
        if args.write_summary:
            summary_writer = tf.summary.FileWriter(train_method_city_dir, sess.graph)

        if restore_path != None and not start_step:   
            variables_to_restore = [v for v in tf.all_variables() 
              if 'Adam' not in v.name and ('feature_extractor' in v.name or 'label_predictor' in v.name)] 
            saver = tf.train.Saver(variables_to_restore, max_to_keep = 8)
            saver.restore(sess, restore_path)
            print('Load model weight from: %s' % restore_path)
        elif restore_path != None:   
            saver = tf.train.Saver(tf.all_variables(), max_to_keep = max_step/save_step)
            saver.restore(sess, restore_path)
            print('Load model weight from: %s' % restore_path)
        
        print('start to run!!') 
        for step in range(max_step):
            
            _mode = 0
            if step % iter_size == 0:
                tic = time.time()
                _mode = 2
            elif step % iter_size == (iter_size-1):
                toc = time.time()
                print ('Processing time for one update: %.1f' % (toc-tic))
                _mode = 1
            
            # read the data for this batch
            images, task_labels, domain_labels = reader.next_train()
            
            _lr = 1e-6 #5e-6
            
            # calculate the weighting of learning rate for feature extractor (f) / domain classifier (d)
            # GA stands for global alignment, CA stands for classwise alignment
            GA_d_weighting = 10.0
            GA_f_weighting = (2. / (1. + np.exp(-10. * float(step)/1200.)) - 1.)  #for ...
            
            CA_f_weighting = 0.1 #0.01
            
            if _CW_Alignment: #jointly optimize as start
                _run_CA = True
            else:
                _run_CA = False
            
            #############################################
            ##################Task Loss##################
            ############################################
            
            if method == 'GA' or method == 'GACA':            
                _, _task_loss, _task_accur = sess.run([train_op_task, task_loss, task_accur],
                                                      feed_dict = {model.rgb: images,
                                                                   model.task_labels: task_labels,
                                                                   model.domain_labels: domain_labels,
                                                                   lr: _lr,
                                                                   mode: _mode})
           
            
            ####################################################
            ##################Global Alignment##################
            ####################################################
            if method == 'GA' or method == 'GACA':            
                _, _GA_loss, _GA_accur, logits_out = sess.run([train_op_GA, GA_domain_loss, GA_domain_accur, model.upsample],
                                              feed_dict = {model.rgb: images,
                                                           model.task_labels: task_labels,
                                                           model.domain_labels: domain_labels,
                                                           lr: (GA_d_weighting * _lr),
                                                           mode: _mode})
                _  = sess.run([train_op_GA_inv],
                              feed_dict = {model.rgb: images,
                                           model.task_labels: task_labels,
                                           model.domain_labels: domain_labels,
                                           lr: (GA_f_weighting * _lr),
                                           mode: _mode})
            #######################################################
            ##################Classwise Alignment##################
            #######################################################
            if _run_CA: 
                _, _CA_loss = sess.run([train_op_CA, cw_loss],  
                            feed_dict = {model.rgb: images,
                                           model.task_labels: task_labels,
                                           model.domain_labels: domain_labels,
                                           lr: (CA_f_weighting * _lr),
                                           mode: _mode})

            if step % 20 == 0:
                format_str_task = ('%s: step %d, task_loss = %.2f, task_accur = %.2f')
                print (format_str_task % (datetime.now(), step, _task_loss, _task_accur))

                print ('Weighting of global alignment:')
                print ('Domain classifier %.3f' % GA_d_weighting)
                print ('Feat. Extractor: %.3f' % GA_f_weighting)
                
                format_str_GA = ('%s: step %d, global_domain_loss = %.3f, global_domain_accur = %.2f')
                print (format_str_GA % (datetime.now(), step, _GA_loss, _GA_accur))

                if _run_CA:
                    print ('Weighting of class-wise alignment:')
                    #print ('Domain classifier %.3f' % CA_d_weighting)
                    print ('Feat. Extractor: %.3f' % CA_f_weighting)
                    format_str_CA = ('%s: step %d, classwise_loss = %.2f')
                    print (format_str_CA % (datetime.now(), step, _CA_loss))
                    
            # Save the model checkpoint periodically.
            if (step != 0 and (step+1) % save_step == 0):
                path = train_method_city_dir + 'model' 
                print ('Saving snapshots to path: %s' % path )
                saver.save(sess, path, global_step = (step+1))

if __name__ == '__main__':
    train()
