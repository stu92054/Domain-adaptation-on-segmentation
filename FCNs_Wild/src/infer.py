import subprocess
import os.path
import argparse
import sys
import os
import numpy as np 
import cv2
import skimage
import skimage.io
import tensorflow as tf

model_path = './models/'
sys.path.insert(0, model_path)
from model import FCN8VGG

parser = argparse.ArgumentParser()
parser.add_argument('--eval_script', type=str, required=True)
parser.add_argument('--city', type=str, required=True)
#parser.add_argument('--load_npy', type=int, required=True)
parser.add_argument('--pretrained_weight', type=str, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--img_dir', type=str, required=False)
parser.add_argument('--img_path_file', type=str, required=False)
parser.add_argument('--weights_dir', type=str, required=False)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--gt_dir', type=str, required=True)
parser.add_argument('--input_width', type=int, default=512)
parser.add_argument('--input_height', type=int, default=256)
parser.add_argument('--_format', type=str, required=True)
parser.add_argument('--gpu', type=str, required=True)
parser.add_argument('--iter_lower', type=int, required=False)
parser.add_argument('--iter_upper', type=int, required=False)
args= parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#if args.load_npy:
#    assert args.method == 'pretrained', "If pretrained is '1', then the method must be 'pretrained'. "
#    print ('Evaluate the model using the pre-trained weight')

#else: 
#assert args.method != 'pretrained', "If pretrained is '0', then the method will be 'GA', 'GACA' or 'Full_method'. "
assert args.img_path_file != None or args.img_dir != None, "At least one input way should be given."
use_pretrained = True
if args.method != 'pretrained':
    use_pretrained = False
    assert args.weights_dir != None, "If pretrained is '0', then the 'weights_dir' must be given according the specific method. "
    assert args.iter_upper >= args.iter_lower and args.iter_lower >= 0, "iter_lower must be larger than iter_upper"

        
    if args.iter_lower == args.iter_upper:
        print ('Evaluate the model at iteration %d...' % (args.iter_lower))
    elif args.iter_upper > args.iter_lower:
        print ('Evaluate the model between iteration %d ~ %d...' % (args.iter_lower, args.iter_upper))


config = tf.ConfigProto( allow_soft_placement = True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config = config)

### build the model
model = FCN8VGG(args.pretrained_weight)

with tf.device( '/gpu:0'):
    with tf.name_scope( "content_vgg"):
        model.build( batch_size=1, train=False, num_classes=19 , city=args.city, debug=False)

init = tf.global_variables_initializer()
sess.run( init)
saver = tf.train.Saver()

### the color of each class in Cityscapes setting ###
###               R   G   B                       ###
road =          [128, 64,128]
sidewalk =      [244, 35,232]
building =      [ 70, 70, 70]
wall =          [102,102,156]
fence =         [190,153,153]
pole =          [153,153,153]
traffic_light = [250,170, 30]
traffic_sign =  [220,220,  0]
vegetation =    [107,142, 35]
terrain =       [152,251,152]
sky =           [ 70,130,180]
person =        [220, 20, 60]
rider =         [255,  0,  0]
car =           [  0,  0,142]
truck =         [  0,  0, 70]
bus =           [  0, 60,100]
train =         [  0, 80,100]
motorcycle =    [  0,  0,230]
bicycle =       [119, 11, 32]

label_colours = np.array([road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign,
                          vegetation, terrain, sky, person, rider, car, truck,
                          bus, train, motorcycle, bicycle])
def generate_image_list( data_path):
    """
    Args:
        data_path: A .txt which has the individual path of all the images.

    Returns:
        A list which which has the individual path of all the images.  
    
    """
    image_list = []
    f = open(data_path,'r')
    for line in f:
        image_list.append(line.split('\n')[0])
    return image_list


def generate_image_list_v2( data_path):
    """
    Args:
        data_path: A folder contains all the images.

    Returns:
        A list which which has the individual path of all the images.  
    
    """
    image_list = []
    for file in os.listdir(data_path):
        image_list.append(os.path.join(data_path, file))
    return image_list


def generate_trained_weights(weights_dir, iter_lower, iter_upper, _format):
    """Generate a list which has all the trained weights in different iteration.
    
    Args:
        weight_dir: A specific directory which contains all the trained weights.
        iter_lower: The lower boundary of iteration.
        iter_upper: The height boundary of iteration. 
        _format: The format of the weight name.
                 [_format-iteration] ex: Tensor("lr:0", shape=(), dtype=float32)-1400       
    
    Returns:
        A list which has all the weights in a specific part from 'iter_lower' to 'iter_upper'
    """
    trained_weights = []
    for _file in sorted(os.listdir( weights_dir)):
        if _format in _file and 'meta' in _file:
            iter_num = int(_file.split('.')[0].split('-')[-1])
            if iter_num >= iter_lower and iter_num <= iter_upper:
                trained_weights.append(_file.split('.')[0])

    assert len(trained_weights) > 0, "File doesn't exist!"
    
    return trained_weights
    

def predict(image_list, _pretrained, output_visualize_dir, output_eval_dir, weight_path = False):
    """Predict the label of each image according to the specific model weight.

    Args:
        image_list: A image list has all images which you want to test.  
        _pretrained: Whether using the pre-trained model weight? 1: True, 0: False
        output_visualize_dir: All the 'predict_visualize' images will be saved at this directory.       
        output_eval_dir: All the 'predict_label' images will be saved at this directory.
        weight_path: If '_pre-trained' is True, you will need the path of the specific model weight. 
                     Default is False, it means that the pre-trained model weight is been adopted.
    """
    if not os.path.isdir( output_visualize_dir):
        os.makedirs( output_visualize_dir)
    if not os.path.isdir( output_eval_dir):
        os.makedirs( output_eval_dir)

    ### restore fintuneed weight or not ###
    if not _pretrained:
        saver.restore( sess, weight_path)
    elif _pretrained and args.city=='syn2real':
        saver.restore( sess, '.pretrained/train_synthia_frontend') 
    
    for i in range( len( image_list)):

        img = skimage.io.imread( image_list[i])
        img = cv2.resize(img, (args.input_width, args.input_height))
        img = np.expand_dims( img, axis=0)
        feed_dict = { model.rgb: img}
        predict_label = sess.run( [ model.pred_up], feed_dict = feed_dict)
        predict_label = np.squeeze( predict_label)

        ### change 19 class to 13 class for our dataset setting ###
        for bld in [3,4,5]: # Building
            predict_label[ predict_label == bld] = 2
        for bus in [14,16]: # Bus
            predict_label[ predict_label == bus] = 15
        for tree in [9]:    # vegetarian       
            predict_label[ predict_label == tree] = 8
    
        predict_rgb = np.zeros(( predict_label.shape[0], 
                      predict_label.shape[1], 3), dtype = np.uint8)
        for cls in range(19):
            predict_rgb[ predict_label==cls] = label_colours[cls] 
         
        predict_bgr = predict_rgb[ :,:,(2,1,0)] 
        ### For visulalization, combine the predict_bgr and the raw image ###
        predict_visualize = ( np.squeeze( img[:,:,:,::-1]) * 0.4 + \
                            predict_bgr * 0.6).astype( np.uint8) 
            
        output_visualize_path = os.path.join(output_visualize_dir, ( image_list[i].split( '/')[-1]).replace( 'jpg', 'png'))
        cv2.imwrite( output_visualize_path, predict_visualize)

        predict_label[np.where(predict_label==18)]=33
        predict_label[np.where(predict_label==17)]=32
        predict_label[np.where(predict_label==16)]=31
        predict_label[np.where(predict_label==15)]=28
        predict_label[np.where(predict_label==14)]=27
        predict_label[np.where(predict_label==13)]=26
        predict_label[np.where(predict_label==12)]=25
        predict_label[np.where(predict_label==11)]=24
        predict_label[np.where(predict_label==10)]=23
        predict_label[np.where(predict_label==9)]=22
        predict_label[np.where(predict_label==8)]=21
        predict_label[np.where(predict_label==7)]=20
        predict_label[np.where(predict_label==6)]=19
        predict_label[np.where(predict_label==5)]=17
        predict_label[np.where(predict_label==4)]=13
        predict_label[np.where(predict_label==3)]=12
        predict_label[np.where(predict_label==2)]=11
        predict_label[np.where(predict_label==1)]=8
        predict_label[np.where(predict_label==0)]=7
     
        output_eval_path = os.path.join(output_eval_dir, ( image_list[i].split( '/')[-1]).replace( 'jpg', 'png'))
        cv2.imwrite( output_eval_path, predict_label)


def evaluation(eval_script, gt_dir, output_dir):
    """Evaluating the 'predict label' against the 'groundtruth label'.
       It will show out the IoU of the each class, and the mIoU of all classes.
        
    Args:
        eval_script: The script of the evluation code which is modified from Cityscapes.
        gt_dir: The directory which has 'groundtruth label' of all the images.
        output_dir: The directory which has 'predict label' of all the images.
    """
    eval_command = 'python ' + eval_script + \
                   ' --gt ' + gt_dir + \
                   ' --pd ' + output_dir
    subprocess.call(eval_command, shell=True)
    
    

if __name__ == '__main__': 
   
    if args.img_dir != None:
        image_path = os.path.join(args.img_dir, args.city)  
    gt_city_dir = os.path.join(args.gt_dir, args.city) 
    
    if args.img_path_file != None:
        image_list = generate_image_list(args.img_path_file)
    else:
        image_list = generate_image_list_v2(image_path)
    
    
    if use_pretrained:
        output_visualize_dir = os.path.join(args.output_dir, args.method, args.city, 'visualize')
        output_eval_dir = os.path.join(args.output_dir, args.method, args.city, 'eval')
        predict( image_list, use_pretrained, 
                     output_visualize_dir, output_eval_dir)
        
        evaluation( args.eval_script, gt_city_dir, output_eval_dir)
        print 'The above results are calculated by using the pre-trained model'

    else:
        weights_method_city_dir = args.weights_dir + args.method + '/' + args.city + '/' 
        trained_weights = generate_trained_weights( weights_method_city_dir, 
                          args.iter_lower, args.iter_upper, args._format)
        
        for trained_weight in trained_weights:
            iteration = trained_weight.split('-')[-1]
            weight_path = os.path.join(weights_method_city_dir, trained_weight)
            output_visualize_dir = os.path.join(args.output_dir, args.method, args.city, 'iter_'+iteration, 'visualize')
            output_eval_dir = os.path.join(args.output_dir, args.method, args.city, 'iter_'+ iteration, 'eval')
                
            predict(image_list, use_pretrained, output_visualize_dir, output_eval_dir, weight_path) 
            
            evaluation( args.eval_script, gt_city_dir, output_eval_dir)
            print 'The above results are calculated by using the model @ iteration %s' % iteration






