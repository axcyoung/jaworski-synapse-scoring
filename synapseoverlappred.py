from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import os, math, random
import tensorflow as tf
print('Tensorflow version ' + tf.__version__)
import numpy as np
import ImageGenerator_3D as generator

#Epochs
EPOCHS = 15
total_loss_list = []

class BCEModel(tf.keras.Model):
    def __init__(self):
        """
        CNN Architecture for Synapse vs. Non-Synapse
        """
        super(BCEModel, self).__init__()

        self.batch_size = 64
        self.num_classes = 1 #because BCE
        self.loss_list = []
        self.learning_rate = .01
        #self.dropout_rate = .3
        #self.var_ep = .000001

        self.layer1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')
        self.layer2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')
        self.dense1 = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape (batch_size, 39[z], 10[y], 10[x], 1[channels])
        :return: probs: shape (batch_size, 1[num_classes])
        """
        layer1Output = self.layer1(inputs)
        layer2Output = self.layer2(layer1Output)
        probs = self.dense1(layer2Output)
        return probs

    def loss(self, probs, labels):
        """
        Calculates the binary cross-entropy loss after one forward pass.
        :param probs: shape (batch_size, 1)
        :param labels: shape (batch_size, 1)
        :return: loss
        """ 
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, probs))
        
    def accuracy(self, probs, labels):
        """
        Calculates the model's prediction accuracy by comparing
        probs to correct labels
        :param probs: shape (batch_size, 1)
        :param labels: shape (batch_size, 1)
        :return: accuracy
        """
        correct_predictions = tf.equal(tf.argmax(probs, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


class CCEModel(tf.keras.Model):
    def __init__(self):
        """
        CNN Architecture for Synapse Overlap Prediction
        """
        super(CCEModel, self).__init__()

        self.batch_size = 64
        self.num_classes = 4
        self.loss_list = []
        self.learning_rate = .01
        #self.dropout_rate = .3
        #self.var_ep = .000001

        self.layer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')
        self.layer2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')
        self.dense1 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape (batch_size, 39[z], 10[y], 10[x], 1[channels])
        :return: probs: shape (batch_size, 4[num_classes])
        """
        layer1Output = self.layer1(inputs)
        layer2Output = self.layer2(layer1Output)
        probs = self.dense1(layer2Output)
        return probs

    def loss(self, probs, labels):
        """
        Calculates the binary cross-entropy loss after one forward pass.
        :param probs: shape (batch_size, 4)
        :param labels: shape (batch_size, 4)
        :return: loss
        """ 
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, probs))
        
    def accuracy(self, probs, labels):
        """
        Calculates the model's prediction accuracy by comparing
        probs to correct labels
        :param probs: shape (batch_size, 4)
        :param labels: shape (batch_size, 4)
        :return: accuracy
        """
        correct_predictions = tf.equal(tf.argmax(probs, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(binary_model, category_model, snsX_train, snsy_train, overlapX_train, overlapy_train):
    '''
    Trains the model on all of the inputs and labels for one epoch. Inputs are batched.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs shape (batch_size, 39[z], 10[y], 10[x], 1[channels])
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    alpha = 0.1
    
    num_examples1 = snsy_train.shape[0]
    shuffle_indices1 = np.arange(0, num_examples1)
    shuffle_indices1 = tf.random.shuffle(shuffle_indices1)
    snsX_train = tf.gather(snsX_train, shuffle_indices1)
    snsy_train = tf.gather(snsy_train, shuffle_indices1)
    
    num_examples2 = overlapy_train.shape[0]
    shuffle_indices2 = np.arange(0, num_examples2)
    shuffle_indices2 = tf.random.shuffle(shuffle_indices2)
    overlapX_train = tf.gather(overlapX_train, shuffle_indices2)
    overlapy_train = tf.gather(overlapy_train, shuffle_indices2)

    optimizer = tf.keras.optimizers.Adam(model.learning_rate)

    for i in range(0, num_examples2, category_model.batch_size):
        snsX_batch = snsX_train[i:i + binary_model.batch_size, :, :, :, :]
        snsy_batch = snsy_train[i:i + binary_model.batch_size, :]
        overlapX_batch = overlapX_train[i:i + category_model.batch_size, :, :, :, :]
        overlapy_batch = overlapy_train[i:i + category_model.batch_size, :, :, :, :]
        
        with tf.GradientTape() as tape:
            B_probs = binary_model.call(snsX_batch)
            synapse_loss = model.loss(B_probs, snsy_batch)
            C_probs = category_model.call(overlapX_batch)
            overlap_loss = category_model.call(C_probs, overlapy_batch)
            #total_loss = synapse_loss + ground_truth_synapse*alpha*overlap_loss

        gradients1 = tape.gradient(total_loss, binary_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients1, binary_model.trainable_variables))
        
        gradients2 = tape.gradient(total_loss, category_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients2, category_model.trainable_variables))
        
        total_loss_list.append(total_loss)
        

def test(model, test_inputs, test_labels, cce=False):
    """
    Tests the model on the test inputs and labels.
    :param model: the initialized model to use for the forward pass and backward pass
    :param test_inputs: test inputs shape (batch_size, 39[z], 10[y], 10[x], 1[channels])
    :param test_labels: test labels (all labels to use for testing), 
    shape (num_labels, num_classes)
    :return: test accuracy - average accuracy across all batches
    """
    accuracies, predicted_labels = [], []
    for i in range(0, test_labels.shape[0], model.batch_size):
        input_batch = test_inputs[i:i + model.batch_size, :, :, :, :]
        label_batch = test_labels[i:i + model.batch_size, :]
        
        probs = model.call(input_batch)
        accuracies.append(model.accuracy(probs, label_batch))
        if cee:
            predicted_labels.append(tf.argmax(probs, 1))
    if cee:
        visualize_results(predicted_labels, tf.argmax(test_labels, 1))
    return np.mean(np.array(accuracies))
    

def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(predicted_labels, true_labels):
    """
    Uses Matplotlib to visualize synapse overlap label accuracy.
    :param predicted_labels
    :param true_labels
    """
    plt.scatter(predicted_labels, true_labels, s=70, alpha=.03)
    plt.title('Predicted v True')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    synnonsyn_input = np.load('data.npy')
    synnonsyn_input = tf.convert_to_tensor(synnonsyn_input, dtype=tf.float32)
    synnonsyn_labels = np.load('synapse_nonsynapse.npy')
    synnonsyn_labels = tf.convert_to_tensor(synnonsyn_labels)
    overlap_input = data[0:191]
    overlap_labels = np.load('labels.npy')
    overlap_labels = tf.convert_to_tensor(labels)
    print('synapse_nonsynapse_input: ' + str(np.shape(synnonsyn_input)))
    print('synapse_nonsynapse_labels: ' + str(np.shape(synnonsyn_labels)))
    print('overlap_input: ' + str(np.shape(overlap_input)))
    print('overlap_labels: ' + str(np.shape(overlap_labels)))
    
    snsX_train, snsX_test, snsy_train, snsy_test = train_test_split(synnonsyn_input, synnonsyn_labels, test_size=0.25, random_state=42)
    overlapX_train, overlapX_test, overlapy_train, overlapy_test = train_test_split(overlap_input, overlap_labels, test_size=0.25, random_state=42)
    
    binary_model = BCEModel()
    category_model = CCEModel()
    for i in range(EPOCHS):
        train(binary_model, category_model, snsX_train, snsy_train, overlapX_train, overlapy_train)
        
    print('Synapse-nonsynapse accuracy: ' + str(test(binary_model, snsX_test, snsy_test)))
    print('Synapse overlap accuracy: ' + str(test(category_model, overlapX_test, overlapy_test, cce=True)))
    visualize_loss(binary_model.loss_list)
    visualize_loss(category_model.loss_list)
    return

if __name__ == '__main__':
    main()
