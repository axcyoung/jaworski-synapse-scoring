from PIL import Image

import os, math, random
import xml.etree.ElementTree as ET
import tensorflow as tf
print('Tensorflow version ' + tf.__version__)
import numpy as np

class Marker:
    def __init__(self, type, X_coord, Y_coord, Z_coord):
        self.type = type
        self.X_coord = X_coord
        self.Y_coord = Y_coord
        self.Z_coord = []
        
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
    for marker in known_markers:
        if marker.type == 1:
            synapse_count += 1
            new_marker = []
            for layer in range(channel1.n_frames):
                try:
                    channel1.seek(layer)
                    channel2.seek(layer)
                    new_layer = []
                    for x in range(marker.X_coord - 5, marker.X_coord + 5):
                        new_X = []
                        for y in range(marker.Y_coord - 5, marker.Y_coord + 5):
                            new_X.append(channel1.getpixel((x, y)) + channel2.getpixel((x, y)))
                        new_layer.append(new_X)
                    new_marker.append(new_layer)
                            
                except EOFError:
                    print("Seek layer error")
                    break
            data.append(new_marker)
    
            closest_marker = None
            closest_distance = float("inf")
            for others in known_markers:
                new_distance = math.sqrt((marker.X_coord - others.X_coord)**2 + (marker.Y_coord - others.Y_coord)**2)
                if new_distance < closest_distance and others.type > 1:
                    closest_marker = others
                    closest_distance = new_distance
            labels.append(closest_marker.type - 2) 
            #fully innervated (2), greater than 50% innervated (3), less than 50% innervated (4), 
            #or fully denervated (5)
    synapse_nonsynapse = [1]*synapse_count
        
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
        if not ((rand_X, rand_Y) in avoid_positives):
            new_marker = []
            for layer in range(channel1.n_frames):
                try:
                    channel1.seek(layer)
                    channel2.seek(layer)
                    new_layer = []
                    for x in range(rand_X - 5, rand_X + 5):
                        new_X = []
                        for y in range(rand_Y - 5, rand_Y + 5):
                            new_X.append(channel1.getpixel((x, y)) + channel2.getpixel((x, y)))
                        new_layer.append(new_X)
                    new_marker.append(new_layer)
                            
                except EOFError:
                    print("Seek layer error")
                    break
            data.append(new_marker)
            nonsynapse_count += 1
    synapse_nonsynapse += [0] * nonsynapse_count
     
    data = np.array(data)
    data = tf.reshape(data, (-1, channel1.n_frames, 10, 10, 1))
    data = tf.cast(data, np.float32)
    data = data/np.float32(255.0) #normalize?

    labels = np.array(labels)
    labels = tf.one_hot(labels, 4, dtype=tf.float32)
    
    synapse_nonsynapse = np.array(synapse_nonsynapse)
    synapse_nonsynapse = tf.one_hot(synapse_nonsynapse, 2, dtype=tf.float32)
    
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