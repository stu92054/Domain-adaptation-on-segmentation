import tensorflow as tf
#from config import *
import numpy as np
import tensorflow as tf
import time
from ccnn import constraintloss
#from tensorflow.python.framework import ops
import pdb

class WeakLoss():
    def __init__(self, src_data_name):
        #self.bg_lower,self.bg_upper = 0.3,0.7
        #self.bg_slack = 1e10        # no slack : 1e10
        #self.fg_lower_hard = 0.01
        #self.fg_lower = 0.05
        self.fg_slack = 2.      # no slack : 1e10
        self.hardness = 1.      # no hardness : 1 and hardness : 1000
        self.dataset = src_data_name

        self.downsample_rate = 4
        self.semi_supervised = False
        self.normalization = True  
                
        # class statistic on source domain
        if self.dataset == 'Cityscapes':   
            # statics for cityscapes
            self.lower_ten = np.asarray([
            2.3576e-01, 2.0294e-03, 4.1489e-02, 0.0000e+00, 0.0000e+00, 1.7319e-03,
            0.0000e+00, 2.2888e-04, 1.3222e-02, 0.0000e+00, 9.1553e-05, 0.0000e+00,
            0.0000e+00, 2.1210e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00]).astype(np.float32)

            self.upper_ten = np.asarray([
            0.4112, 0.1197, 0.3577, 0.0145, 0.0251, 0.023,  0.0062, 0.0115, 0.2917, 0.0299,
            0.0812, 0.0283, 0.0031, 0.1594, 0.0006, 0.,     0.,     0.0009, 0.0098]).astype(np.float32)

            self.avg = np.asarray([
            3.3456e-01, 4.1985e-02, 1.9857e-01, 0.0000e+00, 0.0000e+00, 8.6060e-03,
            1.0681e-04, 2.8458e-03, 1.2714e-01, 3.9673e-04, 2.6543e-02, 2.4872e-03,
            0.0000e+00, 4.0009e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            2.5177e-04]).astype(np.float32)
        else:    
            # statics for synthia
            self.lower_ten = np.asarray([
            1.6006e-02, 4.7562e-02, 1.5131e-01, 0.0000e+00, 0.0000e+00, 2.0676e-03,
            0.0000e+00, 2.2888e-05, 7.4158e-03, 0.0000e+00, 1.1520e-03, 1.0880e-02,
            4.0436e-04, 3.5858e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            1.4496e-04]).astype(np.float32)
            
            self.upper_ten = np.asarray([
            0.3364, 0.3543, 0.4629, 0.0049, 0.0053, 0.0194, 0.001,  0.002,  0.2294, 0.,
            0.1686, 0.0882, 0.0106, 0.1094, 0.,     0.0346, 0.,     0.0051, 0.0052]).astype(np.float32)
            
            self.avg = np.asarray([
            1.8266e-01, 1.8216e-01, 2.7080e-01, 0.0000e+00, 4.0436e-04, 7.5607e-03,
            5.3406e-05, 3.6621e-04, 8.1596e-02, 0.0000e+00, 5.2475e-02, 3.2417e-02,
            2.5482e-03, 2.0409e-02, 0.0000e+00, 2.7695e-03, 0.0000e+00, 5.3406e-04,
            1.2894e-03]).astype(np.float32)

    def calculate(self,x): # forward
        #x is in type numpy 
        D = x.shape[-1]; H = x.shape[1]; W = x.shape[2];
        ds = self.downsample_rate 
        x = x[:, 0:H:ds, 0:W:ds, :] #subsample for coarse output to reduce computing
        batch_size = int(x.shape[0]/2) 
        #bottom = bottom[batch_size:,...] # only get target
        self.diff = []
        loss,w = 0,0
        
        for i in range(batch_size, 2*batch_size): #iter over batch_size

            if (not self.semi_supervised):         # weakly-supervised downsampled training
                # Setup bottoms
                f = np.ascontiguousarray(x[i].reshape((-1,D)))       # f : height*width x channels
                q = np.exp(f-np.max(f,axis=1)[:,None])                              # expAndNormalize across channels
                q/= np.sum(q,axis=1)[:,None]

                # Setup the constraint softmax
                csm = constraintloss.ConstraintSoftmax(self.hardness)
                # calculate image_level label
                pred = np.argmax(x[i], axis=-1)
                pred_avg = np.zeros((D))
                for cla in range(D):
                    pred_avg[cla] = np.mean(pred == cla)
                L = pred_avg > (0.1* self.lower_ten) 
                
                csm.addZeroConstraint( (~L).astype(np.float32) )
                
                # Add Positive label constraints
                for cla in np.flatnonzero(L):
                    #if cla > 6:
                    #    break # early stop for debug
                    v_onehot = np.zeros((D)).astype(np.float32)
                    v_onehot[cla] = 1 
                    csm.addLinearConstraint(  v_onehot, float(self.avg[cla]), self.fg_slack ) # lower bound
                    csm.addLinearConstraint( -v_onehot, float(-self.upper_ten[cla]) ) # upper bound
                
                # Run constrained optimization
                #start_time = time.clock()
                p = csm.compute(f)
                #print('opt time', time.clock()-start_time) 
                
                if self.normalization:
                    self.diff.append( ((q-p).reshape(x[i].shape)) / np.float32(f.shape[0]))      # normalize by (f.shape[0])
                else:
                    self.diff.append( ((q-p).reshape(x[i].shape)) )      # unnormalize
            
            if self.normalization:          
                loss += (np.sum(p*np.log(np.maximum(p,1e-10))) - np.sum(p*np.log(np.maximum(q,1e-10))))/np.float32(f.shape[0])    # normalize by (f.shape[0])
            else:
                loss += (np.sum(p*np.log(np.maximum(p,1e-10))) - np.sum(p*np.log(np.maximum(q,1e-10))))    # unnormalize

        loss /= batch_size

        self.diff = np.array(self.diff)
        #pdb.set_trace()
        self.diff = self.diff.repeat(ds, axis=1).repeat(ds, axis=2)
        ### mapping to original image size
        patch = np.zeros((batch_size,ds,ds,D)).astype(np.float32); patch[:,0,0,:] = 1;
        valid_mask = np.tile(patch, (1,H/ds,W/ds,1))
        self.diff = self.diff * valid_mask
        #pdb.set_trace()
        self.diff = np.concatenate((np.zeros_like(self.diff), self.diff),0)
        
        # process grad for size constrain: reduce by 0.1 for major class
        grad_reduce_map = np.ones(self.diff.shape).astype(np.float32)
        grad_reduce_map[:,:,:,0] = 0.1; grad_reduce_map[:,:,:,2]=0.1; grad_reduce_map[:,:,:,8]=0.1;
        self.diff = self.diff * grad_reduce_map # gradient for backward
        return loss.astype(np.float32)

       
