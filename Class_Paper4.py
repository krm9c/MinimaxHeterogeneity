# The distributed learning paradigm
# Author : Krishnan Raghavan
# Date: June 22nd, 2018

#######################################################################################
# Define all the libraries
import random
import numpy as np
import tensorflow as tf
import operator
from functools import reduce


####################################################################################
# Helper Function for the weight and the bias variable initializations
# Weight
####################################################################################
def sample_Z(m, n, kappa):
    # return(np.random.uniform(-kappa, kappa, size=[m, n]))
    return(np.random.normal(kappa, kappa, size=[m, n]))


############################################################################################
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


#############################################################################################
#  Summaries for the variables
def variable_summaries(var, key):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries' + key):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev' + key):
           stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))


#############################################################################################
def init_ftn(name, num_input, num_output, runiform_range):
    if(name == "normal"):
        return(tf.truncated_normal([num_input, num_output]))
    elif(name == "uniform"):
        return(tf.random_uniform([num_input, num_output], minval=-runiform_range, maxval=runiform_range))
    else:
        print("not normal or uniform")


#############################################################################################
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
        self.Layer =[]
        self.Weights =[]
        self.Zs =[]
        self.Cost =[]
        self.Noise_List =[]


#############################################################################################
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


#############################################################################################
    # Function for optimizing the noise
    def Noise_optimizer(self, lr):
        train_list =[]
        weight, bias = self.Noise_List[0]
        weight_update = -1*tf.gradients(self.classifier["Overall_cost"], weight)[0] + 0.0001*weight
        bias_update   = -1*tf.gradients(self.classifier["Overall_cost"], bias)[0] + 0.0001*bias
        # #Generate the updated variables
        train_list.append((weight_update, weight))
        train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)


#############################################################################################
    def Grad_Descent(self, lr):
        train_list = []
        for i in xrange(len(self.Weights)):
            weight, bias = self.Weights[i]
            ## Gradient Descent update
            weight_update = tf.gradients(self.classifier['Overall_cost'], weight)[0]   + 0.0001*weight
            bias_update   = tf.gradients(self.classifier['Overall_cost'],   bias)[0]   + 0.0001*bias
            # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)


#############################################################################################
    def Grad_No_Noise_Descent(self, lr):
        train_list = []
        for i in xrange(len(self.Weights)):
            weight, bias = self.Weights[i]
            ## Gradient Descent update
            weight_update = tf.gradients(self.classifier["Overall_cost_Grad"], weight)[0]   + 0.0001*weight
            bias_update   = tf.gradients(self.classifier["Overall_cost_Grad"],   bias)[0]   + 0.0001*bias
            # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)

#############################################################################################
    def Def_Network(self, classes, Layers, act_function, batch_size, back_range = 1):
        with tf.name_scope("Trainer_Network"):

            # The noise transfoirmations and the setup.
            # Start with defining the affine transformation
            self.classifier['A'] = weight_variable([Layers[0], Layers[0]], trainable=False, name='A')
            self.classifier['b'] = bias_variable([Layers[0]], trainable=False, name='b')
            # Append the affine transformaiton into the noise list  
            self.Noise_List.append((self.classifier['A'], self.classifier['b']))

            # The first layer stuff
            # Defining the weight and bias variable for the first layer
            self.classifier['WeightFL_layer_11'] = weight_variable([Layers[0], Layers[1]], trainable=False, name='WeightFL_layer_11')
            self.classifier['BiasFL_layer_11']   = bias_variable([Layers[1]], trainable=False, name='BiasFL_layer_11')
            # The input noise model
            input_model     = tf.matmul(self.Deep['FL_layer_10'], self.classifier['A']) + self.classifier['b']
            input_model     = tf.nn.l2_normalize(input_model)


            
            preactivate     = tf.matmul(input_model, self.classifier['WeightFL_layer_11']) + self.classifier['BiasFL_layer_11']
            self.Deep\
            ['FL_layer_11'] = act_function(preactivate)
            self.Weights.append((self.classifier['WeightFL_layer_11'], self.classifier['BiasFL_layer_11']))
            
            # The rest of the layers.
            for i in range(2, len(Layers)):
                key = 'FL_layer_1' + str(i)
                self.Deep['FL_layer_1' + str(i)]\
                = self.nn_layer(self.Deep['FL_layer_1' + str(i-1)], Layers[i-1],
                Layers[i], act=act_function, trainability=False, key=key)
                # weights for each layer is append to the training list
                self.Weights.append((self.classifier['Weight'+key], self.classifier['Bias'+key]))
        
        # The final classification layer
        with tf.name_scope("Classifier"):
            self.classifier['class_Noise']=self.nn_layer(self.Deep['FL_layer_1' + str(len(Layers) - 1)],
            Layers[len(Layers)-1], classes, act=tf.identity, trainability=False, key='class')
            # weights for the final layer is appended on to the training list
            self.Weights.append((self.classifier['Weight'+'class'], self.classifier['Bias'+'class']))
        
        # Finally, we define an additional network for producing noise free outputs
        ############ The network without the noise, primarily for output estimation
        self.Deep['FL_layer_3' + str(0)] = self.Deep['FL_layer_10']
        for i in range(1, len(Layers)):
            key = 'FL_layer_1' + str(i)
            preactivate = tf.matmul(self.Deep['FL_layer_3' + str(i - 1)],
                                    self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            self.Deep['FL_layer_3' +str(i)] = act_function(preactivate, name='activation_3' + key)
        
        self.classifier['class_NoNoise'] = tf.identity(tf.matmul(self.Deep['FL_layer_3' + str(len(Layers) - 1)],\
        self.classifier['Weightclass']) + self.classifier['Biasclass'])


 ############################################################################################################################
    def init_NN_custom(self, classes, lr, Layers, act_function, batch_size,
                        back_range,  par='GDR', act_par="tf.nn.tanh"):
        ########################### Initial Definitions
            with tf.name_scope("PlaceHolders"):  
                #### Setup the placeholders        
                # Label placeholder
                self.classifier['Target'] = tf.placeholder(
                tf.float32, shape=[None, classes])

                # Place holder for the learning rate
                self.classifier["learning_rate"] = tf.placeholder(
                tf.float32, [], name='learning_rate')
                
                # Input placeholder
                self.Deep['FL_layer_10'] = tf.placeholder(
                tf.float32, shape=[None, Layers[0]])


        ########################################################## The network
                # The main network
                self.Def_Network(classes, Layers, act_function, batch_size, back_range)
        

        ################################################# Design the trainer for the  network
            with tf.name_scope("Trainer"):
            #The overall cost function
                ## Direct Costs
                self.classifier["cost_NN"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.classifier['class_Noise'], labels=self.classifier['Target'], name='Error_Cost')) 
                self.classifier["cost_NN_nonoise"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.classifier['class_NoNoise'], labels=self.classifier['Target'], name='Error_Cost_Nonoise')) 
                self.classifier["Cost_M1"] = 0.99*self.classifier["cost_NN"] + 0.01*self.classifier["cost_NN_nonoise"]
                self.classifier["L2Cost"] = tf.nn.l2_loss(self.classifier['A'])+tf.nn.l2_loss(self.classifier['b'])

                ## KL divergence
                Dist_1 = tf.nn.softmax(self.classifier['class_NoNoise'])
                Dist_2 = tf.nn.softmax(self.classifier['class_Noise'])
                self.classifier["Entropy"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Dist_1, labels=Dist_1, name='Entropy')) 
                self.classifier["Cross_Entropy"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
                (logits=Dist_1, labels=Dist_2, name='CrossEntropy')) 
                self.classifier["KL"] = (np.square(0.1))*(self.classifier["Cross_Entropy"])
                
                # The final cost function
                self.classifier["Overall_cost"] = self.classifier["Cost_M1"]-self.classifier["KL"]
                self.classifier["Overall_cost_Grad"] = self.classifier["cost_NN_nonoise"]


        ##########################################################################################################
        # The final optimization
            with tf.name_scope("Trainers"): 
                # Call the other optimizer
                self.Trainer["Grad_op"]  = self. Grad_Descent(self.classifier["learning_rate"])
                self.Trainer["Grad_No_Noise_op"]  = self.Grad_No_Noise_Descent(self.classifier["learning_rate"])
                self.Trainer["Noise_op"] = self.Noise_optimizer(self.classifier["learning_rate"])
                

        ############## The evaluation section of the methodology
            with tf.name_scope('Evaluation'):
                self.Evaluation['correct_prediction'] = \
                        tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class_NoNoise']),1 ) , 
                                tf.argmax(self.classifier['Target'], 1))
                self.Evaluation['accuracy'] = tf.reduce_mean(
                        tf.cast(self.Evaluation['correct_prediction'], tf.float32))
            self.sess.run(tf.global_variables_initializer())


            return self




