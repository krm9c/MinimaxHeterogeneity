##### The test file
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import Class_Paper_CBD_minmax as NN_class
import tensorflow as tf
import numpy as np
import traceback
import gzip, cPickle

from sklearn import preprocessing
import math
import gc
from random import random



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
def return_dict(model, batch_x, batch_y, lr):
    S={}
    S[model.Deep['FL_layer_10']] = batch_x
    S[model.classifier['Target']] = batch_y
    S[model.classifier["learning_rate"]] = lr
    return S

####################################################################################
def sample_Z(X, m, n, kappa):
    return (X+np.random.uniform(-kappa, kappa, size=[m, n]))
    # return(X+np.random.normal(0, kappa, size=[m, n]))

####################################################################################
def Analyse_custom_Optimizer_GDR_old(X_train, y_train, X_test, y_test, kappa, gamma, batch_size):
    import gc
    print("classes", classes)
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    depth = []
    depth.append(X_train.shape[1])
    L = [100 for i in xrange(5)]
    depth.extend(L)
    lr = 0.001
    model           = model.init_NN_custom(classes, lr, depth, tf.nn.relu, batch_size, gamma)
    acc_array       = np.zeros((Train_Glob_Iterations, 1))
    acc_array_train = np.zeros((Train_Glob_Iterations, 1))
    cost_M1         = np.zeros((Train_Glob_Iterations, 1))
    cost_M2         = np.zeros((Train_Glob_Iterations, 1))
    cost_M3         = np.zeros((Train_Glob_Iterations, 1))
    cost_Total      = np.zeros((Train_Glob_Iterations, 1))

    T = sample_Z(X_train, X_train.shape[0], X_train.shape[1], kappa=1)
    
    import random as random 
    
    try:
        t = xrange(Train_Glob_Iterations)
    
        from tqdm import tqdm
        for i in tqdm(t):
            lr   = 0.99*lr
            lr_N = lr 
            ########### Batch learning update
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys = batch
                _ = model.sess.run([model.Trainer['EDL_op']],\
                feed_dict=return_dict(model, batch_xs, batch_ys, lr))

                # Train the noise.
                lr_N=0.99*lr_N
                _ = model.sess.run([model.Trainer['EDL_Noise_op']],\
                feed_dict=return_dict(model, batch_xs, batch_ys, lr_N))


            ########## Evaluation portion
            if i % 1 == 0:
                # Evaluation and display part
                acc_array[i] = model.sess.run([ model.Evaluation['accuracy']], \
                feed_dict={model.Deep['FL_layer_10']: T, model.classifier['Target']: \
                y_train, model.classifier["learning_rate"]:lr})
                
                acc_array_train[i] = model.sess.run([ model.Evaluation['accuracy']],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']:\
                y_train, model.classifier["learning_rate"]:lr})
                
                cost_Total[i] = model.sess.run([model.classifier["Overall_cost"]],\
                feed_dict={model.Deep['FL_layer_10']: batch_xs, model.classifier['Target']:\
                batch_ys,model.classifier["learning_rate"]:lr})
                
                cost_M1[i]  = model.sess.run([ model.classifier["Cost_M1"] ],\
                feed_dict={model.Deep['FL_layer_10']: batch_xs, model.classifier['Target']:\
                batch_ys, model.classifier["learning_rate"]:lr})    
                
                cost_M2[i]  = model.sess.run([ model.classifier["KL"] ],\
                feed_dict={model.Deep['FL_layer_10']: batch_xs, model.classifier['Target']:\
                batch_ys, model.classifier["learning_rate"]:lr})


                # Print all the outputs
                if(i % 1 == 0):
                    print("---------------------------------------------------------------------------------------------------------")
                    print("Accuracies", i, "With Noise", acc_array[i],"Without Noise", acc_array_train[i],)
                    print("Overall Cost", cost_Total[i], "Cost_M1", cost_M1[i], "Cost_M2 KL", cost_M2[i], "Cost_M3 Z cost", cost_M3[i])
                    print("---------------------------------------------------------------------------------------------------------")

                # ################################################################################            
                # Stop the learning in case of this condition being satisfied
                if max(acc_array) > 0.99:
                    break

        acc_array       = np.zeros((len(kappa), 1))
        acc_array_train = np.zeros((len(kappa), 1))
        
        for i,element in enumerate(kappa):
            Noise_data = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa=element)
            acc_array[i] = model.sess.run([ model.Evaluation['accuracy']], \
            feed_dict={model.Deep['FL_layer_10']:Noise_data, model.classifier['Target']: y_test})
            acc_array_train[i] = model.sess.run([ model.Evaluation['accuracy']],\
            feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']: y_train})

    
    except Exception as e:
        print("I found an exception", e)
        traceback.print_exc()
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0

    tf.reset_default_graph()
    gc.collect()
    
    return np.reshape(acc_array, (len(kappa))),\
    np.reshape(acc_array_train, (len(kappa))),\
    np.reshape( (acc_array-acc_array_train), (len(kappa)) ),\
    np.reshape( (cost_Total), (len(kappa)) ),\
    np.reshape( (cost_M1), (len(kappa) ) ),\
    np.reshape( (cost_M2), (len(kappa))  ) 



path_here = '/usr/local/home/krm9c/Documents/Research/Parallel/shwetaNew/data/'
datasets = ['arcene', 'cifar10', 'cifar100', 'gas', 'gisette', 'madelon',\
'mnist', 'notmnist', 'rolling', 'sensorless', 'SVHN']
def load_data(datasetName):
    print datasetName
    f = gzip.open(path_here+datasetName+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    ''' try:
        if X_train.shape[2]:
            print X_train.shape, y_train.shape, X_test.shape, y_test.shape
            X_train = X_train.reshape((X_train.shape[0],(X_train.shape[1]*X_train.shape[2]*X_train.shape[3])))
            X_test = X_test.reshape((X_test.shape[0],(X_test.shape[1]*X_test.shape[2]*X_test.shape[3])))
    except IndexError:
        pass
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape '''
    return X_train, y_train, X_test, y_test


N = 1
for i, element in enumerate(datasets):
    print("elements", element)
    X_train, y_train, X_test, y_test  = load_data(element)
    ### Flatten the data and scale it 
    if element in [ 'cifar10', 'cifar100', 'mnist', 'notmnist', 'SVHN']:
        d1, _, _, _ = X_train.shape
        X_train= X_train.reshape((d1, -1)) 
        d1, _, _, _ = X_test.shape
        X_test= X_test.reshape((d1, -1))
    else:
        d1, _ = X_train.shape
        X_train= X_train.reshape((d1, -1)) 
        d1, _ = X_test.shape
        X_test= X_test.reshape((d1, -1))
    scaler = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Main results.
    Gamma = np.random.uniform(0, 1, size=[1])

    from tqdm import tqdm

    for t in tqdm(xrange(len(Gamma))):
        gamma = Gamma[t]
        # Setup the parameters and call the functions
        Train_batch_size = 64
        Train_Glob_Iterations = 50
        Train_noise_Iterations = 1

        from tqdm import tqdm
        from tensorflow.examples.tutorials.mnist import input_data

        classes = y_train.shape[1]
        # mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
        # X_train = mnist.train.images
        # X_test = mnist.test.images
        # y_train = mnist.train.labels
        # y_test = mnist.test.labels

        print("Train", X_train.shape, "Test", X_test.shape)
        inputs = X_train.shape[1]
        filename =str(element)+'_Relu_5_Uniform_Noise_alpha.csv'
        iterat_kappa = Train_Glob_Iterations 
        Kappa_s = np.random.uniform(0, 1, size=[iterat_kappa])

        Results = np.zeros([iterat_kappa, 7])
        Results[:, 0] = Kappa_s
        Results[:, 1], Results[:, 2], Results[:, 3],         \
        Results[:, 4],  Results[:, 5],  Results[:, 6]        \
        =Analyse_custom_Optimizer_GDR_old(X_train, y_train, \
        X_test, y_test, Kappa_s, gamma, batch_size=Train_batch_size)
        np.savetxt(filename, Results, delimiter=',')
