
# Testing of Generalized Convolutional Neural Network

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import tensorflow as tf
import numpy as np
import gzip, cPickle, random
import cnn_class as NN_class

# Getting the data
path_here ='/usr/local/home/krm9c/Documents/Research/Parallel/shwetaNew/data/' 
datasets = ['arcene', 'cifar10', 'cifar100', 'gas', 'gisette', 'madelon',\
'mnist', 'notmnist', 'rolling', 'sensorless', 'SVHN']

datasetName = 'mnist'

def load_data(datasetName):
    print datasetName
    f = gzip.open(path_here+datasetName+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]


    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    num_channels = 1
    img_size = []
    try:
        if X_train.shape[2]:
            img_size = (X_train.shape[1],X_train.shape[2])
            num_channels = X_train.shape[3]
            X_train = X_train.reshape((X_train.shape[0],(X_train.shape[1]*X_train.shape[2]*X_train.shape[3])))
            X_test = X_test.reshape((X_test.shape[0],(X_test.shape[1]*X_test.shape[2]*X_test.shape[3])))
    except IndexError:
        pass

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    return X_train, y_train, X_test, y_test, img_size, num_channels

X_train, y_train, X_test, y_test, img_size, num_channels = load_data(datasetName)


# Activation Function
def act_ftn(name):
    if(name == "tanh"):
        return(tf.nn.tanh)
    elif(name == "relu"):
        return(tf.nn.relu)
    elif(name == 'sigmoid'):
        return(tf.nn.sigmoid)
    else:
        print("no activation")

def perturbData(X, m, n, kappa):
    return (X+np.random.normal(kappa, kappa, size=[m, n]))
    #return (X+(10*np.random.uniform(-4,4,size=[m,n])))


# Model parameters
img_size_flat = X_train.shape[1]
img_shape = img_size
classes = y_train.shape[1]

steps = 501
batch_size = 50
display_step = 20
learning_rate = 0.001
optimizer = 'Adam'

num_conv_layers = 2
num_fc_layers = 2

# Convolutional Layer 1
filter1_size = 5
number_of_filter1 = 16

# Convolutional Layer 2
filter2_size = 5
number_of_filter2 = 32

# Max Pooling shape
pool_shape = [3,3]

# Fully Connected Layer 
fc_size = 128

filter_size = []
filter_size.append(filter1_size)
filter_size.append(filter2_size)

num_of_filters = []
num_of_filters.append(num_channels)
num_of_filters.append(number_of_filter1)
num_of_filters.append(number_of_filter2)

# Define model
model = NN_class.learners()
model = model.init_NN_custom(classes, num_conv_layers, num_fc_layers, img_size_flat, \
    img_size, num_channels, filter_size, num_of_filters, pool_shape, fc_size, learning_rate, optimizer=optimizer)


# Training
for j in range(steps):

    for k in xrange(100):
        x_batch = []
        y_batch = []
        arr = random.sample(range(0, len(X_train)), batch_size)
        for idx in arr:
            x_batch.append(X_train[idx])
            y_batch.append(y_train[idx])
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        model.sess.run([model.Trainer["Weight_op"]],\
        feed_dict={model.Deep['Conv_Layer_00']: x_batch, model.classifier['Target']: \
        y_batch, model.classifier["learning_rate"]:learning_rate})

    if j%display_step == 0:        
        print "Step", j
        #X_test_perturbed = perturbData(X_test, X_test.shape[0], X_test.shape[1])
        
        acc_test = model.sess.run([model.Evaluation['accuracy']], \
        feed_dict={model.Deep['Conv_Layer_00']: X_test, model.classifier['Target']:\
        y_test, model.classifier["learning_rate"]:learning_rate})
        
        acc_train = model.sess.run([ model.Evaluation['accuracy']],\
        feed_dict={model.Deep['Conv_Layer_00']: X_train, model.classifier['Target']:\
            y_train, model.classifier["learning_rate"]:learning_rate})

        cost_error = model.sess.run([model.classifier["Overall cost"]],\
        feed_dict={model.Deep['Conv_Layer_00']: x_batch, model.classifier['Target']:\
        y_batch, model.classifier["learning_rate"]:learning_rate})

        # Print all the outputs
        print("Loss: ", cost_error[0])
        print("Train Acc:", acc_train[0]*100, "Test Acc:", acc_test[0]*100)

n_iters = 400
Kappa = np.random.uniform(0, 4, size=[n_iters])
acc_array = np.zeros((2,len(Kappa)))
print(Kappa.shape)
for i,element in enumerate(Kappa):
        print(i)
        Noise_data =  perturbData(X_test, X_test.shape[0], X_test.shape[1], kappa = element)
        acc_array[0,i] = np.reshape(model.sess.run([model.Evaluation['accuracy']], \
        feed_dict={model.Deep['Conv_Layer_00']: Noise_data, model.classifier['Target']:\
        y_test, model.classifier["learning_rate"]:learning_rate}), [1])
        acc_array[1,i] = element

np.savetxt("Noise_Acc_MNIST_CNN_NoAugmentedTraining.csv", acc_array, delimiter=',')

''' print("Final Test Accuracy", model.sess.run([ model.Evaluation['accuracy']], \
        feed_dict={model.Deep['Conv_Layer_00']: X_test, model.classifier['Target']: \
        y_test, model.classifier["learning_rate"]:learning_rate})[0]*100) '''