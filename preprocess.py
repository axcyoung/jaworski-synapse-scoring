from PIL import Image

import os, math, random
import xml.etree.ElementTree as ET
import tensorflow as tf
print('Tensorflow version ' + tf.__version__)
import numpy as np

#Builds tensor of max pixel values (taking max of the two channels) with shape (n, 39, 10, 10, 1)
#(39, 10, 10, 1) windows around each of the n synapses
#Builds tensor of synapse-nonsynapse labels with shape (n)
#Builds one-hot tensor of overlap scoring labels with shape (n, 4)
#Consider capturing fewer than the full 39-layer stack, as each synapse tends to only span a few layers

class Marker:
    def __init__(self, type, X_coord, Y_coord, Z_coord):
        self.type = type
        self.X_coord = X_coord
        self.Y_coord = Y_coord
        self.Z_coord = []
        #self.best_Z = 0
        #self.best_f = 0.0
        
def get_data(c_0, c_1, xml_markers):
    known_markers = []
    data = []
    labels = []
    synapse_nonsynapse = []

    tree = ET.parse(xml_markers)
    root = tree.getroot()
    for markertype in root[1]:
        for marker in markertype:
            if markertype.find('Type') != None and marker.find('MarkerX') != None:
                known_markers.append(Marker(int(markertype.find('Type').text), 
                                           int(marker.find('MarkerX').text),
                                           int(marker.find('MarkerY').text),
                                           1))
                                           
    channel1 = Image.open(c_0)
    channel2 = Image.open(c_1)
    
    #Synapses
    synapse_count = 0
    type_2 = 0
    type_3 = 0
    type_4 = 0
    type_5 = 0
    for marker in known_markers:
        if marker.type == 1:
            synapse_count += 1
            new_marker = []
            for layer in range(channel1.n_frames):
                try:
                    channel1.seek(layer)
                    channel2.seek(layer)
                    #current_f = 0.0
                    new_layer = []
                    for x in range(marker.X_coord - 5, marker.X_coord + 5):
                        new_X = []
                        for y in range(marker.Y_coord - 5, marker.Y_coord + 5):
                            #current_f += max((channel1.getpixel((x, y)), channel2.getpixel((x, y))))
                            new_X.append(max((channel1.getpixel((x, y)), channel2.getpixel((x, y)))))
                        new_layer.append(new_X)
                    new_marker.append(new_layer)
                    #if current_f > marker.best_f:
                    #    marker.best_f = current_f
                    #    marker.best_Z = layer
                            
                except EOFError:
                    print("Seek layer error")
                    break
            data.append(new_marker)
            #if marker.best_Z < 5:
            #    data.append(new_marker[0:10])
            #elif marker.best_Z > channel1.n_frames - 5:
            #    data.append(new_marker[channel1.n_frames-10:channel1.n_frames])
            #else:
            #    data.append(new_marker[marker.best_Z - 5:marker.best_Z + 5])
    
            closest_marker = None
            closest_distance = float("inf")
            for others in known_markers:
                new_distance = math.sqrt((marker.X_coord - others.X_coord)**2 + (marker.Y_coord - others.Y_coord)**2)
                if new_distance < closest_distance and others.type != 1:
                    closest_marker = others
                    closest_distance = new_distance
            if closest_marker.type == 2:
                type_2 += 1
            elif closest_marker.type == 3:
                type_3 += 1
            elif closest_marker.type == 4:
                type_4 += 1
            else:
                type_5 += 1
            labels.append(closest_marker.type - 2) 
            #fully innervated (2), greater than 50% innervated (3), less than 50% innervated (4), 
            #or fully denervated (5)
    labels += [3]*synapse_count #filler so that size matches?
    synapse_nonsynapse = [1]*synapse_count
    print("TYPES")
    print(type_2)
    print(type_3)
    print(type_4)
    print(type_5)
        
    #Build negatives
    avoid_positives = []
    for marker in known_markers:
        if marker.type == 1:
            positive_X = marker.X_coord
            positive_Y = marker.Y_coord
            for x in range(positive_X - 20, positive_X + 20):
                for y in range(positive_Y - 20, positive_Y + 20):
                    avoid_positives.append((x, y))
    
    nonsynapse_count = 0
    while nonsynapse_count < synapse_count:
        rand_X = random.randint(6, channel1.size[0] - 6) 
        rand_Y = random.randint(6, channel1.size[1] - 6)
        #rand_Z = random.randint(6, channel1.n_frames - 6)
        if not ((rand_X, rand_Y) in avoid_positives):
            new_marker = []
            for layer in range(channel1.n_frames): #otherwise in rand_Z range
                try:
                    channel1.seek(layer)
                    channel2.seek(layer)
                    new_layer = []
                    for x in range(rand_X - 5, rand_X + 5):
                        new_X = []
                        for y in range(rand_Y - 5, rand_Y + 5):
                            new_X.append(max((channel1.getpixel((x, y)), channel2.getpixel((x, y)))))
                        new_layer.append(new_X)
                    new_marker.append(new_layer)
                            
                except EOFError:
                    print("Seek layer error")
                    break
            data.append(new_marker)
            nonsynapse_count += 1
    synapse_nonsynapse += [0]*nonsynapse_count
     
    data = np.array(data)
    data = tf.reshape(data, (-1, 39, 10, 10, 1))
    data = tf.cast(data, np.float32)
    #data = data/np.float32(255.0) #normalize?

    labels = np.array(labels)
    labels = tf.one_hot(labels, 4, dtype=tf.float32)
    
    synapse_nonsynapse = np.array(synapse_nonsynapse)
    #synapse_nonsynapse = tf.one_hot(synapse_nonsynapse, 1, dtype=tf.float32)
    
    print('datasize: ' + str(np.shape(data)))
    print(data[0:10])
    print('labelsize: ' + str(np.shape(labels)))
    print(labels[0:10])
    print('synapse_nonsynapsesize: ' + str(np.shape(synapse_nonsynapse)))
    print(synapse_nonsynapse[0:10])
    
    np.save('data.npy', data, allow_pickle=False)
    np.save('labels.npy', labels, allow_pickle=False)
    np.save('synapse_nonsynapse.npy', synapse_nonsynapse, allow_pickle=False)

if __name__ == '__main__':
    get_data('13.oib - C=0.tif', '13.oib - C=1.tif', 'CellCounter_MAX_13.xml')