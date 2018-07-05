# The test file
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import Class_Paper4 as NN_class
import tensorflow as tf
import numpy as np
import traceback
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
from sklearn.preprocessing import normalize
def sample_Z(X, m, n, kappa):
    # return (X+np.random.uniform(-kappa, kappa, size=[m, n]))
    # A = np.random.normal(0, kappa, size=[n, n])
    # b = np.random.normal(0, kappa, size=[n])
    # A = np.random.uniform(-kappa, kappa, size=[n, n])
    # b = np.random.uniform(-kappa, kappa, size=[n])
    # print("Norm while estimating", np.linalg.norm(A)+np.linalg.norm(b), np.linalg.norm((np.dot(X,A)+ b)))
    # return normalize((np.dot(X,A)+b))

    # return X+np.random.uniform(-kappa, kappa, size=[m, n])
    return X+np.random.normal(0, kappa, size=[m, n])

####################################################################################
def Analyse_custom_Optimizer_GDR_old(X_train, y_train, X_test, y_test, kappa, batch_size,back_range):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    depth = []
    depth.append(X_train.shape[1])
    L = [100 for i in xrange(3)]
    depth.extend(L)
    lr = 0.001
    lr_N = 0.01
    updates         = 1
    model           = model.init_NN_custom(classes, lr, depth, tf.nn.relu, batch_size, back_range)
    acc_array       = np.zeros((Train_Glob_Iterations, 1))
    acc_array_train = np.zeros((Train_Glob_Iterations, 1))
    cost_M1         = np.zeros((Train_Glob_Iterations, 1))
    cost_M2         = np.zeros((Train_Glob_Iterations, 1))
    cost_Balance    = np.zeros((Train_Glob_Iterations, 1))
    T = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa=1)
    import random as random
    try:
        t = xrange(Train_Glob_Iterations/10)
        from tqdm import tqdm
        for i in tqdm(t):
            lr   = lr
            lr_N = 100*lr
            ########### Batch learning update
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys = batch
                rand = random.random()
                
                # # Train the main network.
                # _ = model.sess.run([model.Trainer["Grad_No_Noise_op"]],\
                # feed_dict=return_dict(model, batch_xs, batch_ys, lr))

                # ## Train the main network.
                _ = model.sess.run([model.Trainer["Grad_op"]],\
                    feed_dict=return_dict(model, batch_xs, batch_ys, lr))

                # lr_N = 1.01*lr
                ## Train the noise.
                _ = model.sess.run([model.Trainer["Noise_op"]],\
                feed_dict=return_dict(model, batch_xs, batch_ys, lr_N))

            ########## Evaluation portion
            if i % 1 == 0:

                # Evaluation and display part
                acc_array[i] = model.sess.run([ model.Evaluation['accuracy']], \
                feed_dict={model.Deep['FL_layer_10']: T, model.classifier['Target']: \
                y_test, model.classifier["learning_rate"]:lr})
                acc_array_train[i] = model.sess.run([ model.Evaluation['accuracy']],\
                feed_dict={model.Deep['FL_layer_10']: X_test, model.classifier['Target']:\
                y_test, model.classifier["learning_rate"]:lr})

                cost_Balance[i] = model.sess.run([model.classifier["Overall_cost"]],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']:\
                y_train,model.classifier["learning_rate"]:lr})
                cost_M2[i]  = model.sess.run([ model.classifier["KL"] ],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']:\
                 y_train, model.classifier["learning_rate"]:lr})
                cost_M1[i]  = model.sess.run([ model.classifier["Cost_M1"] ],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']:\
                y_train, model.classifier["learning_rate"]:lr})    
                entropies  = model.sess.run([ model.classifier["Cross_Entropy"], model.classifier["Entropy"], model.classifier["L2Cost"]],\
                feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']:\
                y_train, model.classifier["learning_rate"]:lr})

                # Print all the outputs
                print("---------------------------------------------------------------------------------------------------------")
                print("Accuracies", i, "With noise", acc_array[i],"Without Noise", acc_array_train[i],)
                print("Cost_M1", cost_M1[i], "Cost_M2", cost_M2[i],"Overall Cost", cost_Balance[i])
                print("Entropies", entropies)
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
    np.reshape( (cost_Balance), (len(kappa)) ),\
    np.reshape( (cost_M1), (len(kappa) ) ),\
    np.reshape( (cost_M2), (len(kappa))  ) 

# Setup the parameters and call the functions
Train_batch_size = 64
back_range = 1
Train_Glob_Iterations = 100
Train_noise_Iterations = 1
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

classes = 10
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels

print("Train", X_train.shape, "Test", X_test.shape)
inputs       = X_train.shape[1]
filename     = 'paper_4_Noise_Training_Gauss_1_relu_5_affine.csv'
iterat_kappa = Train_Glob_Iterations 
Kappa_s = np.random.uniform(0, 1, size=[iterat_kappa])
Results = np.zeros([iterat_kappa, 7])
Results[:, 0] = Kappa_s

Results[:, 1], Results[:, 2], Results[:, 3],  \
Results[:, 4],  Results[:, 5],  Results[:, 6] \
= Analyse_custom_Optimizer_GDR_old(
X_train, y_train, X_test, y_test, Kappa_s, batch_size=Train_batch_size, back_range=back_range)
np.savetxt(filename, Results, delimiter=',')
