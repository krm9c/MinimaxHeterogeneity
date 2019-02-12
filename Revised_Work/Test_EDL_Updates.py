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
def Analyse_custom_Optimizer_GDR_old(X_train, y_train, X_test, y_test):
    import gc
    model = NN_class.learners()
    depth = []
    depth.append(X_train.shape[1])

    # parameter set 
    B_choice = 1  # actual choice of range of B(random uniform distribution)
    kappa    = 1  # choice of the decay coefficient
    N_hidden   = 5  # number of hidden layers + 2
    lr = 0.001 # Learning rate
    act =  tf.nn.tanh # The activation funciton
    par_GRAD_EDL = 0  # 0 for EDL and 1 for GRAD
    classes = 10 # The number of classes.
    alpha = 0.1 ## parameter controlling the impact of neighborhood 
    act_par ="tf.nn.tanh"


    # Things to return for logging
    Cost = np.zeros((Train_Glob_Iterations,1))
    Train_acc = np.zeros((Train_Glob_Iterations,1))
    Test_acc  = np.zeros((Train_Glob_Iterations,1))
    # Create the NN array
    L = [100 for i in xrange(N_hidden)]
    depth.extend(L)
    # Call the initialization function.
    model = model.init_NN_custom( classes, lr, depth, act_function= act, batch_size=128,\
    back_range= B_choice, kappa = kappa, par='EDL', act_par = act_par)
    try:
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        for i in tqdm(t):
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys = batch
                batch_xs_pert =sample_Z(batch_xs, batch_xs.shape[0], batch_xs.shape[1], kappa=1)

                train_x_batch = np.concatenate((batch_xs, batch_xs_pert))
                train_y_batch = np.concatenate((batch_ys, batch_ys))
                
                model.sess.run([model.Trainer["EDL_op"]], feed_dict={model.Deep['FL_layer_10']: train_x_batch, model.classifier['Target']: \
                train_y_batch, model.classifier["learning_rate"]:lr})

            if i%20== 0:        
                X_test_perturbed = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa=1)
                print( "Accuracies", model.sess.run([model.Evaluation['accuracy']], \
                feed_dict={model.Deep['FL_layer_10']: X_test_perturbed, model.classifier['Target']:\
                y_test, model.classifier["learning_rate"]:lr}), model.sess.run([ model.Evaluation['accuracy']],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']: y_train}) )

            if i%1== 0:      
                # Get into arrays
                Train_acc[i] = model.sess.run([ model.Evaluation['accuracy']],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']: y_train}) 
                Test_acc[i] = model.sess.run([ model.Evaluation['accuracy']],\
                feed_dict={model.Deep['FL_layer_10']: X_test, model.classifier['Target']: y_test}) 
                Cost[i] = model.sess.run([ model.classifier["cost_NN"] ],\
                feed_dict={model.Deep['FL_layer_10']: X_test, model.classifier['Target']: y_test}) 

        iterate_kappa = len(Cost)
        Kappa = np.random.uniform(0, 4, size=[iterate_kappa])
        acc_array = np.zeros((len(Kappa),1))
        for i,element in enumerate(Kappa):
                Noise_data = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa=element)
                
                acc_array[i] = model.sess.run([ model.Evaluation['accuracy']], \
                feed_dict={model.Deep['FL_layer_10']:Noise_data, model.classifier['Target']: y_test})
        
        return np.reshape(Train_acc, (iterate_kappa)), np.reshape(Test_acc, (iterate_kappa)),\
        np.reshape(Cost, (iterate_kappa)), np.reshape(acc_array, (iterate_kappa)) 
    
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
Train_batch_size = 128
Train_Glob_Iterations  = 50
Repeat_Steps = 100

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

Tr_acc = np.zeros((Repeat_Steps,Train_Glob_Iterations))
T_acc = np.zeros((Repeat_Steps,Train_Glob_Iterations))
Tot_Cost = np.zeros((Repeat_Steps,Train_Glob_Iterations))
Noise_acc = np.zeros((Repeat_Steps,Train_Glob_Iterations))

for j in tqdm(xrange(Repeat_Steps)):
    classes = 10
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels
    y_test = mnist.test.labels
    Tr_acc[j,:], T_acc[j,:], Tot_Cost[j,:], Noise_acc[j,:] = Analyse_custom_Optimizer_GDR_old(X_train, y_train, X_test, y_test)

np.savetxt("Train_Accuracy_MNIST.csv", Tr_acc, delimiter=',')
np.savetxt("Test_Accuracy_MNIST.csv", T_acc, delimiter=',')
np.savetxt("Tot_Cost_MNIST.csv", Tot_Cost, delimiter=',')
np.savetxt("Noise_Acc_MNIST.csv", Noise_acc, delimiter=',')