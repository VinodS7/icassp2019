from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np
import time

import inputs
import utils_tf

from scipy.io import wavfile as wav



class CarliniWagnerAttack():

    def __init__(self,model,save_model_dir,sess,hparams):
        self.hparams = hparams
        self.save_model_dir = save_model_dir
        self.sess = sess
        self.model = model
        return

    def build_attack(self):
        
        """
        Create graph for the adversarial attacks

        """
        hparams = self.hparams
        #Input audio shape
        self.delta_shape = tf.placeholder(tf.int32,name='qq_delta_init')
        
        #Delta is the perturbation to be added to the input to create ther adversarial attack
        self.delta = delta =  tf.Variable(tf.zeros(shape=self.delta_shape),validate_shape=False,name='qq_delta')
        
        self.mask = tf.Variable(tf.zeros(shape=self.delta_shape),validate_shape=False,name='qq_mask')
        self.labels = tf.placeholder(tf.int32,name='qq_labels')
        self.const = tf.Variable(hparams.const,name='qq_const')
        
        self.apply_delta = tf.clip_by_value(delta,-1.0,1.0)*self.mask

        self.original = tf.Variable(tf.zeros(shape=self.delta_shape),validate_shape=False,name='qq_original')
        self.new_input = new_input = self.apply_delta + self.original
        features = inputs.compute_features(self.new_input,hparams)
         
        features.set_shape([None,25,64]) 
        self.output = self.model.get_logits(features)
        self.probs = tf.nn.softmax(self.output)
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])

        saver.restore(self.sess,self.save_model_dir)

        self.l2dist = tf.reduce_mean(tf.square(self.new_input-self.original))
        self.label = tf.one_hot(self.labels,41)
        num_examples = tf.shape(features)[0]
        self.label = tf.tile([self.label],[num_examples, 1])
        self.label = tf.cast(self.label,tf.float32)
        self.real = real = tf.reduce_sum((self.label)*self.output,1)
        self.other = other = tf.reduce_max((1-self.label)*self.output-self.label*10000,1)
        if hparams.targeted:
            loss1 = tf.maximum(0.,other-real+hparams.confidence)
        else:
            loss1 = tf.maximum(0.,real-other+hparams.confidence)
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2
        
        start_vars = set(x.name for x in tf.global_variables())
        self.optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
        self.train = self.optimizer.minimize(self.loss, var_list=[self.delta])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.init = tf.variables_initializer(var_list=[self.delta]+new_vars)           
        return
    
   
    def attack(self,input_audio,labels,is_first=True,offset=10):
        #Initialize all variables. Load session for simplicity
        hparams = self.hparams
        sess = self.sess
        audio_shape = input_audio.shape #Need input audio shape to initialize variables
        sess.run([self.init,tf.variables_initializer([self.const,self.original,self.delta,self.mask])],feed_dict={self.delta_shape:audio_shape})
        sess.run(self.original.assign(input_audio))
        mask = np.zeros(audio_shape)
        ind = np.argwhere(input_audio>1e-5)
        mask[ind] = 1
        print(mask)
        sess.run(self.mask.assign(mask))
        sess.run(tf.variables_initializer(self.optimizer.variables()),feed_dict={self.delta_shape:input_audio.shape})
        
        final_delta = [None]

        #now = time.time()
        MAX = hparams.max_iterations
        i = 0
        while(True):
            now = time.time()
            l1,l2,l,op,ad=sess.run([self.real,self.loss2,self.other,self.probs,self.new_input],feed_dict={'qq_labels:0':labels}) 
            sig = np.mean(np.square(input_audio))
            l2 = np.squeeze(l2)
            l2 = 10*np.log10(sig/l2)
            l1 = np.squeeze(l1)
            op = np.squeeze(op)
            ad = np.squeeze(ad) 
            if(op.ndim!=1):
                op = np.mean(op,axis=0)
            if(i==0):
                op_orig = op
            elif(i==MAX or l2<15):
                if(hparams.targeted):
                    if(np.argmax(op) == labels):
                        return ad,op,op_orig,l2
                    else:
                        return None,None,None,None
                else:
                    if(np.argmax(op)!=labels):
                        return ad,op,op_orig,l2
                    else:
                        return None,None,None,None
            print(l2,np.argmax(op),np.max(op),labels,op[labels],'\r')
            if(hparams.targeted):
                if(np.argmax(op) == labels and op[labels]>0.6):
                    snr = l2
                    return ad,op,op_orig,snr
            else:
                if(np.argmax(op) != labels and op[labels]<0.05):
                    snr=l2
                    return ad,op,snr
            sess.run([self.train],feed_dict={'qq_labels:0':labels})
            i+=1
        return input_audio


