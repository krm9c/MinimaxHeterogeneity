# The test file
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import Class_Recheck_vEDL as NN_class
import tensorflow as tf
import numpy as np
import traceback
import random
###################################################################################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

###################################################################################
def return_dict(placeholder, List, model, batch_x, batch_y, lr):
    S ={}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer_10']]         = batch_x
    S[model.classifier['Target'] ]       = batch_y
    S[model.classifier["learning_rate"]] = lr
    return S

####################################################################################
def sample_Z(X, m, n, kappa):
    return(X+np.random.uniform(-kappa, kappa, size=[m, n]))
    # return ((X+np.random.normal(0,kappa, size=[m, n])))

####################################################################################
def Analyse_custom_Optimizer_GDR_old(X_train, y_train, X_test, y_test, kappa):
    import gc
    model = NN_class.learners()
    depth = []
    depth.append(X_train.shape[1])
    L = [100 for i in xrange(3)]
    depth.extend(L)
    lr = 0.001
    model = model.init_NN_custom(classes, lr, depth, tf.nn.relu)
    try:
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        for i in tqdm(t):
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys = batch
                batch_xs_pert =sample_Z(batch_xs, batch_xs.shape[0], batch_xs.shape[1], kappa=1)

                train_x_batch = np.concatenate((batch_xs, batch_xs_pert))
                train_y_batch = np.concatenate((batch_ys, batch_ys))
                
                model.sess.run([model.Trainer["Weight_op"]], feed_dict={model.Deep['FL_layer_10']: train_x_batch, model.classifier['Target']: \
                train_y_batch, model.classifier["learning_rate"]:lr})

            if i%20== 0:        
                print "Step", i
                X_test_perturbed = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa=1)
                print( "Accuracies", model.sess.run([model.Evaluation['accuracy']], \
                feed_dict={model.Deep['FL_layer_10']: X_test_perturbed, model.classifier['Target']:\
                y_test, model.classifier["learning_rate"]:lr}), model.sess.run([ model.Evaluation['accuracy']],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']: y_train}) )
            
    except Exception as e:
        print("I found an exception", e)
        traceback.print_exc()
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0

#######################################################################################################
################################ Parameters and function call##########################################
#######################################################################################################
# Setup the parameters and call the functions
Train_batch_size = 64
Train_Glob_Iterations  = 501
Train_noise_Iterations = 1

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
classes = 10
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels

iterat_kappa = 1
Kappa_s = np.random.uniform(0, 1, size=[iterat_kappa])
Analyse_custom_Optimizer_GDR_old(X_train, y_train, X_test, y_test, Kappa_s[0])