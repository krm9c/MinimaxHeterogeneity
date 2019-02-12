# This is an example of using Tensorflow to build Sparse Autoencoder
# for representation learning.
# It is the implementation of the sparse autoencoder for
#        https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
#
# For any enquiry, please contact Dr. Zhiwei Lin  at Ulster University
#       http://scm.ulster.ac.uk/zhiwei.lin/
#
#
# ==============================================================================
import tensorflow as tf
import matplotlib.pyplot
import math

# The test file
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import tensorflow as tf
import numpy as np
import traceback

###################################################################################
def return_dict(placeholder, List, model, batch_x, batch_y, lr):
    S = {}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer_10']] = batch_x
    S[model.classifier['Target']] = batch_y
    S[model.classifier["learning_rate"]] = lr 
    return S

####################################################################################
def sample_Z(X, m, n, kappa):
    return(X+np.random.uniform(-kappa, kappa, size=[m, n]))
    # return ((X+np.random.normal(0,kappa, size=[m, n])))

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

####################################################################################
# Helper Function for the weight and the bias variable initializations
# Weight
####################################################################################


def xavier(fan_in, fan_out):
   # use 4 for sigmoid, 1 for tanh activation
   low = -1 * np.sqrt(1.0 / (fan_in + fan_out))
   high = 1 * np.sqrt(1.0 / (fan_in + fan_out))
   return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)
############################################################################################
def weight_variable(shape, trainable, name):
   initial = xavier(shape[0], shape[1])
   return tf.Variable(initial, trainable=trainable, name=name)


#############################################################################################
# Bias function
def bias_variable(shape, trainable, name):
   initial = tf.random_normal(shape, trainable, stddev=1)
   return tf.Variable(initial, trainable=trainable, name=name)

class FeedforwardSparseAutoEncoder():
    '''
      This is the implementation of the sparse autoencoder for https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
    '''
    def __init__(self, n_input, n_hidden, n_classes,  lr = 0.01, rho=0.01, alpha=0.001, beta=3, activation=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer()):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.rho=rho  # sparse parameters
        self.alpha =alpha
        self.lr = lr
        self.beta=beta
        self.optimizer=optimizer
        self.activation = activation
        self.classifier  = {}

        self.X=tf.placeholder("float",shape=[None,n_input])
        self.y=tf.placeholder("float",shape=[None,self.n_classes])

        self.W1=self.init_weights([self.n_input,self.n_hidden])
        self.b1=self.init_weights([self.n_hidden])

        self.W2=self.init_weights([self.n_hidden,self.n_input])
        self.b2= self.init_weights([self.n_input])

        self.W3= self.init_weights([self.n_hidden, self.n_classes])
        self.b3= self.init_weights([self.n_classes])
        self.sess = tf.Session()

    def init_weights(self,shape):
        r= math.sqrt(6) / math.sqrt(self.n_input + self.n_hidden + 1)
        weights = tf.random_normal(shape, stddev=r)
        return tf.Variable(weights)

    def encode(self,X):
        l=tf.matmul(X, self.W1)+self.b1
        return self.activation(l)

    def decode(self,H):
        l=tf.matmul(H,self.W2)+self.b2
        return self.activation(l)

    def classify(self,C):
        l=tf.matmul(C,self.W3)+self.b3
        return l

    def kl_divergence(self, rho, rho_hat):
        return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

    def regularization(self,weights):
        return tf.nn.l2_loss(weights)

    def loss(self):
        H = self.encode(self.X)
        rho_hat=tf.reduce_mean(H,axis=0)   #Average hidden layer over all data points in X, Page 14 in https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
        kl=self.kl_divergence(self.rho, rho_hat)
        X_=self.decode(H)
        diff=self.X-X_
        self.classifier['output']  =  self.classify(H)


        # Sparse Autoencoder
        # cost=   0.5*tf.reduce_mean(tf.reduce_sum(diff**2,axis=1))  \
        #       + 0.5*self.alpha*(tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2))   \
        #       + self.beta*tf.reduce_sum(kl)\
        #       + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #         logits=self.classifier['output'], labels=self.y, name='Error_Cost'))

        # Denoising Autoencoder
        cost=   0.5*tf.reduce_mean(tf.reduce_sum(diff**2,axis=1))  \
              + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.classifier['output'], labels=self.y, name='Error_Cost'))
        return cost

    def training(self, n_iter=1000):
        var_list=[self.W1,self.W2, self.W3, self.b1, self.b2, self.b3]
        loss_=self.loss()
        train_list =[]
        for i,weight in enumerate(var_list):
            ## Gradient Descent update
            weight_update = tf.gradients(loss_, weight)[0] + 0.001*weight
            # Generate the updated variables
            train_list.append((weight_update, weight))
        return tf.train.AdamOptimizer(self.lr).apply_gradients(train_list)
    
    def Model_setup(self):
        self.classifier["training"] = self.training()
        self.classifier['correct_prediction'] = tf.equal(tf.argmax(tf.nn.softmax(self.classifier['output']),1 ), tf.argmax(self.y, 1))
        self.classifier['accuracy'] = tf.reduce_mean(tf.cast(self.classifier['correct_prediction'], tf.float32))
        self.sess.run(tf.global_variables_initializer())


def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    n_inputs=784
    n_hidden=100
    n_classes =10
    n_batch = 64
    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels
    y_test = mnist.test.labels
    sae=   FeedforwardSparseAutoEncoder(n_inputs,n_hidden, n_classes)
    sae.Model_setup()
    n_iters=400
    t = xrange(n_iters)
    for j in xrange(n_iters):
        for batch in iterate_minibatches(X_train, y_train, n_batch, shuffle=True):
            batch_xs, batch_ys = batch
            _ = sae.sess.run([sae.classifier["training"]],feed_dict={sae.X: batch_xs, sae.y: batch_ys})

        if j % 1 ==0:
            print("i,  ", j, ",  Accuracy, ", sae.sess.run([sae.classifier["accuracy"]],feed_dict={sae.X: X_train, sae.y: y_train}))
    Kappa = np.random.uniform(0, 4, size=[n_iters])
    acc_array = np.zeros((2,len(Kappa)))
    print(Kappa.shape)
    for i,element in enumerate(Kappa):
            Noise_data = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa=element)
            acc_array[0,i] = element
            acc_array[1,i] = np.reshape(sae.sess.run([sae.classifier["accuracy"]],feed_dict={sae.X: Noise_data, sae.y: y_test}), [1])
            

    np.savetxt("Noise_Acc_MNIST_Denoising_Autoencoder.csv", acc_array, delimiter=',')

if __name__=='__main__':
    main()