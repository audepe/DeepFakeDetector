import logging
import numpy as np
import json
import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """Multi-threaded data generator.
    Each thread read a batch of images and their object labels

    # Arguments:
        dictionary (dict): Dictionary of image filenames and object labels
        shufle (bool): If dataset should be shuffled before sampling
    """

    def __init__(self,
                 batch_size,
                 dictionary,
                 shuffle=True):

        self.batch_size = batch_size
        self.dictionary = dictionary
        self.keys = list(self.dictionary.keys())

        # data augmentation

        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch
        """
        batch_len = np.floor(len(self.dictionary) / self.batch_size)
        return int(batch_len)

    def __getitem__(self, index):
        """Get a batch of data"""
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size

        keys = self.keys[start_index:end_index]

        x, y = self.__data_generation(keys)
        return x, y

    def on_epoch_end(self):
        """Shuffle after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.keys)

    def __data_generation(self, keys):
        """Generate train data: images and
        object detection ground truth labels

        # Arguments:
            keys (array): Randomly sampled keys (key is image filename)
        # Returns:
            x (tensor): Batch images
            y (tensor): Batch  classes, offsets and masks
        """

        y = np.zeros(len(keys))

        sequences = []


        for i, key in enumerate(keys):
            sequence = []
            files = [x for x in Path(key).iterdir() if x.is_file()]
            files = sorted(files, key=lambda x: int(
                ''.join(list(filter(str.isdigit, str(Path(x).stem))))))
            for file in files:
                sequence.append(np.asarray(load_img(str(file))))
            sequences.append(np.asarray(sequence))
            y[i] = self.dictionary[key]

        x = np.asarray(sequences)
        return x, y

def main():
    with open("sequences.json", "r") as read_file:
        sequences = json.load(read_file)

    data_generator = DataGenerator(2,sequences)
    
    for x,y in data_generator:
        print(len(x))
        print("##############################")
        print(len(y))

if __name__ == "__main__":
    main()
