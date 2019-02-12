# Generalized Feed Forward Neural Network

import tensorflow as tf
import numpy as np
import random
import operator

# Xavier Initialization
def xavier(fan_in, fan_out):
    # use 4 for sigmoid, 1 for tanh activation
    low = -1 * np.sqrt(1.0 / (fan_in + fan_out))
    high = 1 * np.sqrt(1.0 / (fan_in + fan_out))
    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)

# Weights Initialization
def weight_variable(shape, trainable, name):
    initial = xavier(shape[0], shape[1])
    return tf.Variable(initial, trainable=trainable, name=name)

# Bias Initialization
def bias_variable(shape, trainable, name):
    initial = tf.random_normal(shape, trainable, stddev=1)
    return tf.Variable(initial, trainable=trainable, name=name)

# The main Class
class learners():
    def __init__(self):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Evaluation = {}
        self.keys = []
        self.sess = tf.Session()
        # Extra Parameters
        self.Weights =[]


    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act,
                 trainability, key, act_par="tf.nn.tanh"):
        with tf.name_scope(key):
            with tf.name_scope('weights' + key):
                self.classifier['Weight' + key] = weight_variable(
                    [input_dim, output_dim], trainable=trainability, name='Weight' + key)
            with tf.name_scope('bias' + key):
                self.classifier['Bias' + key] = bias_variable(
                    [output_dim], trainable=trainability, name='Bias' + key)
            with tf.name_scope('Wx_plus_b' + key):
                preactivate = tf.matmul(
                    input_tensor, self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            return act(preactivate, name='activation' + key)

    def Custom_Optimizer(self, lr, optimizer):
        train_list = []
        for i in xrange(len(self.Weights)):
            weight, bias = self.Weights[i]
            ## Gradient Descent update
            weight_update = tf.gradients(self.classifier['Overall cost'], weight)[0]   + 0.0001*weight
            bias_update   = tf.gradients(self.classifier['Overall cost'],   bias)[0]   + 0.0001*bias
            # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(0.01).apply_gradients(train_list)

    # Function for defining the network
    def Def_Network(self, classes, Layers, act_function, batch_size, back_range = 1):
        with tf.name_scope("Trainer_Network"):
            self.classifier['WeightFL_layer_11'] = weight_variable([Layers[0], Layers[1]], trainable=False, name='WeightFL_layer_11')
            self.classifier['BiasFL_layer_11'] = bias_variable([Layers[1]], trainable=False, name='BiasFL_layer_11')
            input_model   = self.Deep['FL_layer_10'] 
            preactivate = tf.matmul(input_model, self.classifier['WeightFL_layer_11']) + self.classifier['BiasFL_layer_11']
            self.Deep['FL_layer_11'] = act_function(preactivate)
            #Extra Variables
            #weights
            self.Weights.append((self.classifier['WeightFL_layer_11'], self.classifier['BiasFL_layer_11']))
            #Neural network for the rest of the layers.
            for i in range(2, len(Layers)):
                key = 'FL_layer_1' + str(i)
                self.Deep['FL_layer_1' + str(i)] = self.nn_layer(self.Deep['FL_layer_1' + str(i-1)], Layers[i-1],\
                Layers[i], act=act_function, trainability=False, key=key)
                #Extra Variables
                #weights
                self.Weights.append((self.classifier['Weight'+key], self.classifier['Bias'+key])) 
        with tf.name_scope("Classifier"):
            self.classifier['class'] = self.nn_layer(self.Deep['FL_layer_1' + str(len(Layers) - 1)],
            Layers[len(Layers)-1], classes, act=tf.identity, trainability=False, key='class')
            self.Weights.append((self.classifier['Weight'+'class'], self.classifier['Bias'+'class']))
        
    def init_NN_custom(self, classes, lr, Layers, act_function, batch_size=64, back_range = 1, optimizer="Adam"):
            with tf.name_scope("PlaceHolders"):  
                # Setup the placeholders
                self.classifier['Target']        = tf.placeholder(tf.float32, shape=[None, classes])
                self.classifier["learning_rate"] = tf.placeholder(tf.float32, [], name='learning_rate')
                self.Deep['FL_layer_10']         = tf.placeholder(tf.float32, shape=[None, Layers[0]])
                # The main network
                self.Def_Network(classes, Layers, act_function, batch_size, back_range)
            with tf.name_scope("Trainer"):
                # The overall cost function
                self.classifier["Overall cost"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.classifier['class'], labels=self.classifier['Target'], name='Error_Cost'))
            # The final optimization
            with tf.name_scope("Trainers"): 
                # Call the other optimizer
                self.Trainer["Weight_op"] = self.Custom_Optimizer(self.classifier["learning_rate"], optimizer)
            # The evaluation section of the methodology
            with tf.name_scope('Evaluation'):
                self.Evaluation['correct_prediction'] = \
                        tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class']),1 ) , 
                                tf.argmax(self.classifier['Target'], 1))
                self.Evaluation['accuracy'] = tf.reduce_mean(
                        tf.cast(self.Evaluation['correct_prediction'], tf.float32))
            self.sess.run(tf.global_variables_initializer())
            return self