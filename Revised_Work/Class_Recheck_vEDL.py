# Learning with an adversary
# Author : Krishnan Raghavan
# Date: May 5th, 2018

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
    return(np.random.uniform(-kappa, kappa, size=[m, n]))


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
def drelu(x):
    zero = tf.zeros(x.get_shape())
    one = tf.ones(x.get_shape())
    return(tf.where(tf.greater(x, zero), one, zero))

#############################################################################################
def dtanh(x):
    return(1 - tf.multiply(tf.nn.tanh(x), tf.nn.tanh(x)))

#############################################################################################
def dsigmoid(x):
    return(tf.multiply((1 - tf.nn.sigmoid(x)), tf.nn.sigmoid(x)))

#############################################################################################
def act_ftn(name):
    if(name == "tanh"):
        return(tf.nn.tanh)
    elif(name == "relu"):
        return(tf.nn.relu)
    elif(name == 'sigmoid'):
        return(tf.nn.sigmoid)
    else:
        print("not tanh or relu")

#############################################################################################
def dact_ftn(name):
    if(name == "tf.nn.tanh"):
        return(dtanh)
    elif(name == "tf.nn.relu"):
        return(drelu)
    elif(name == "tf.nn.sigmoid"):
        return(dsigmoid)
    else:
        print("not tanh or relu")

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
        self.Layer_cost =[]
        self.Noise_List =[]
        self.Evaluation = {}
        self.keys = []
        self.sess = tf.Session()

#############################################################################################
    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act, trainability, key, act_par):
        with tf.name_scope(key):
            with tf.name_scope('weights' + key):
                self.classifier['Weight' + key] = weight_variable([input_dim, output_dim], trainable=trainability, name='Weight' + key)
            with tf.name_scope('bias' + key):
                self.classifier['Bias' + key] = bias_variable(
                    [output_dim], trainable=trainability, name='Bias' + key)
            with tf.name_scope('Wx_plus_b' + key):
                preactivate = tf.matmul(
                    input_tensor, self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            activations = act(preactivate, name='activation' + key)
            if (act_par == "tf.nn.tanh"):
                return activations, tf.matrix_diag(dtanh(preactivate) + 0.001)
            elif(act_par == "tf.nn.relu"):
                return activations, tf.matrix_diag(drelu(preactivate) + 0.001)
            elif(act_par == "tf.nn.sigmoid"):
                return activations, tf.matrix_diag(dsigmoid(preactivate) + 0.001)

##################################################################################
    def Optimizer(self, lr, dError_dy, kappa):
        train_list = []
        a = []
        var_list = []
        self.classifier['real_cost'] = []
        self.classifier['updates'] = []
        for i in xrange(len(self.Layer_cost)):
            cost, weight, bias, fan_in, batch_size, layer_in, dlayer,\
                backward = self.Layer_cost[i]
            
            if i < (len(self.Layer_cost) - 1):
                cost = self.Deep['FL_layer_1' + str(i+1)]
                temp_mul = tf.matmul(backward, dlayer)
                dError_dhidden = tf.matmul(dError_dy, temp_mul)

                reshaped_layer_in = tf.reshape(
                    layer_in, [batch_size, fan_in, 1])

                upd_weight = tf.reduce_mean(
                    tf.matmul(reshaped_layer_in, dError_dhidden), 0)

                upd_bias = tf.reduce_mean(dError_dhidden, 0)

                ### Calculate the decay components
                layer_out = cost+ tf.random_normal(tf.shape(cost), mean= 0.0, stddev = 0.001)
                s,_,_    = tf.svd( layer_out )
                diag_mat = tf.sqrt( tf.abs(tf.linalg.diag(s))+0.001);
                sum_diag = (tf.reduce_sum(diag_mat)+0.01)
                div_diag_mat = tf.truediv(diag_mat, sum_diag, name=None)
                # Reshape back to output dimensions and then get the gradients.
                fac =  tf.subtract(tf.eye(tf.shape(diag_mat)[1]),  0.1*div_diag_mat)
                weight_update = 0.5*upd_weight  + 0.5*tf.gradients(self.classifier['Overall cost'], weight)[0]+ 0.0001*tf.matmul(weight, fac)
                bias_update   = 0.5*upd_bias[0] + 0.5*tf.gradients(self.classifier['Overall cost'], bias)[0]  + 0.0001*tf.squeeze( tf.matmul( fac,tf.expand_dims(bias,1) ), axis = 1)
            else:
                reshaped_layer_in = tf.reshape(
                    layer_in, [batch_size, fan_in, 1])

                dError_dhidden = tf.matmul(dError_dy, dlayer)

                upd_weight = tf.reduce_mean(
                    tf.matmul(reshaped_layer_in, dError_dhidden), 0)

                upd_bias = tf.reduce_mean(dError_dy, 0)

                weight_update = upd_weight   + 0.5*tf.gradients(self.classifier['Overall cost'], weight)[0] + 0.0001*weight
                bias_update  = upd_bias[0]   + 0.5*tf.gradients(self.classifier['Overall cost'],  bias )[0] + 0.0001*bias
                
           # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)

    def Custom_Optimizer(self, lr):
        train_list = []
        for i in xrange(len(self.Layer_cost)):
            cost, weight, bias, fan_in, \
            batch_size, layer_in, dlayer, backward = self.Layer_cost[i]

            ## Gradient Descent update
            weight_update = tf.gradients(self.classifier['Overall cost'], weight)[0] + 0.001*weight
            bias_update   = tf.gradients(self.classifier['Overall cost'], bias)[0]   + 0.001*bias

            # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)


#############################################################################################
    def Def_Network(self, classes, Layers, act_function, batch_size, back_range,  act_par):
        i =1
        with tf.name_scope("Trainer_Network"):
            self.classifier['WeightFL_layer_11'] = weight_variable(
            [Layers[i - 1], Layers[i]], trainable=False, name='WeightFL_layer_11')
            self.classifier['BiasFL_layer_11'] = bias_variable([Layers[i]], trainable=False, name='BiasFL_layer_11')
            # Actual First layer
            preactivate = tf.matmul(
            self.Deep['FL_layer_10'] , self.classifier['WeightFL_layer_11']) + self.classifier['BiasFL_layer_11']
            self.Deep['FL_layer_11'] = act_function(preactivate)
            # The first layer derivative
            self.Deep['dFL_layer_11'] = tf.matrix_diag(dtanh(preactivate) + 0.001)
            print("Layers in the NN ", Layers)
            # Neural network for the rest of the layers.
            for i in range(2, len(Layers)):
                # nn_layer(self, input_tensor, input_dim, output_dim, act, trainability, key):
                print("Layers in the NN ", Layers[i], Layers[i-1])
                print("Current Layer", self.Deep['FL_layer_1' + str(i-1)])

                self.Deep['FL_layer_1' + str(i)], self.Deep['dFL_layer_1' + str(i)] = self.nn_layer(self.Deep['FL_layer_1' + str(i-1)], Layers[i-1], Layers[i],\
                act= act_function, trainability=False, key='FL_layer_1' + str(i), act_par =act_par)

                fan_in = Layers[i-2]
                fan_out = Layers[i-1]
                num_final = classes
                cost_layer = self.Deep['FL_layer_1' + str(i)]
                weight_temp = self.classifier['Weight' + 'FL_layer_1' + str(i-1)]
                bias_temp = self.classifier['Bias' + 'FL_layer_1' + str(i - 1)]
                layer_in = self.Deep['FL_layer_1' + str(i - 2)]
                dlayer = self.Deep['dFL_layer_1' + str(i - 1)]

                backward_t = tf.Variable(init_ftn("uniform", num_final, fan_out, back_range))
                backward = tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [batch_size, num_final, fan_out])
                
                self.Layer_cost.append((cost_layer, weight_temp, bias_temp, fan_in, batch_size, layer_in, dlayer, backward))
                

##################################################### Layer d-1 #######################################################################
#################### Layer wise information for generating the updates
        with tf.name_scope("Classifier"):
            self.classifier['class_Noise'], self.classifier['dclass_1']=self.nn_layer(self.Deep['FL_layer_1' + str(len(Layers) - 1)],
            Layers[len(Layers)-1], classes, act=tf.identity, trainability=False, key='class', act_par =act_par)

##################################################### Layer d-1 #######################################################################
#################### Layer wise information for generating the updates
            fan_in      =   Layers[len(Layers) - 2]
            fan_out     =   Layers[len(Layers) - 1]
            num_final   =   classes
            cost_layer  =   self.classifier['class_Noise']
            weight_temp =   self.classifier['Weight' + 'FL_layer_1' + str(len(Layers) - 1)]
            bias_temp   =   self.classifier['Bias'   + 'FL_layer_1' + str(len(Layers) - 1)]
            layer_in    =   self.Deep['FL_layer_1' + str(len(Layers) - 2)]
            dlayer      =   self.Deep['dFL_layer_1' + str(len(Layers) - 1)]
            backward_t  =   tf.Variable(init_ftn("uniform", num_final, fan_out, back_range))
            backward    =   tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                            batch_size, num_final, fan_out ])
            self.Layer_cost.append((cost_layer, weight_temp, bias_temp,
            fan_in, batch_size, layer_in, dlayer, backward))


    def init_NN_custom(self, classes, lr, Layers, act_function, batch_size, back_range,   kappa, par, act_par):
            i = 1
            num_final = classes
            batch_size = 2*batch_size
        ########################### Initial Definitions
            with tf.name_scope("FLearners_1"):  
                # Setup the placeholders
                self.classifier['Target'] = tf.placeholder(
                tf.float32, shape=[None, classes])
                self.classifier["learning_rate"] = tf.placeholder(
                tf.float32, [], name='learning_rate')
                self.Deep['FL_layer_10'] = tf.placeholder(
                tf.float32, shape=[None, Layers[0]])

        ########################################################## The network
                self.Def_Network(classes, Layers, act_function, batch_size, back_range, act_par)
        ################################################# Design the trainer for the  network
            with tf.name_scope("Trainer"):
                # Cross Entropy loss 
                self.classifier["cost_NN"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.classifier['class_Noise'], labels=self.classifier['Target'], name='Error_Cost'))
                fan_in      =   Layers[len(Layers) - 1]
                fan_out     =   classes
                num_final   =   classes
                cost_layer  =   self.classifier["cost_NN"]
                weight_temp =   self.classifier['Weightclass']
                bias_temp   =   self.classifier['Biasclass']
                layer_in    =   self.Deep['FL_layer_1' + str(len(Layers) - 1)]
                backward_t  =   tf.Variable(tf.eye(classes))
                dlayer      =   tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                                batch_size, num_final, num_final ])
                backward_t  =   tf.Variable(init_ftn("uniform", num_final, fan_out, back_range))
                backward    =   tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                                batch_size, num_final, fan_out ])
                self.Layer_cost.append(
                (cost_layer, weight_temp, bias_temp,
                fan_in, batch_size, layer_in, dlayer, backward))
                self.classifier["Overall cost"] = self.classifier["cost_NN"]
        #########################################################################################################
        # The final optimization
            with tf.name_scope("Trainers"): 
                # Call the other optimizer
                self.Trainer["Grad_op"] = self.Custom_Optimizer(self.classifier["learning_rate"])
                # The general optimizer
                dError_dy = tf.reshape(tf.gradients(self.classifier["Overall cost"], self.classifier['class_Noise'])[0], [batch_size, 1, num_final])
                # the optimizers
                self.Trainer['EDL_op'] = self.Optimizer(self.classifier["learning_rate"],  dError_dy, kappa)
        ############## The evaluation section of the methodology
            with tf.name_scope('Evaluation'):
                self.Evaluation['correct_prediction'] = \
                        tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class_Noise']),1 ) , 
                                tf.argmax(self.classifier['Target'], 1))
                self.Evaluation['accuracy'] = tf.reduce_mean(
                        tf.cast(self.Evaluation['correct_prediction'], tf.float32))
            self.sess.run(tf.global_variables_initializer())
            return self




