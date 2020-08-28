"""
Dataset loading

"""

import torch.utils.data as Data
import os
import h5py
import numpy as np
import csv


class guidancedatasetnew(Data.Dataset):
    """
    A simple dataset that will take medical images from a folder and provide image
    and metadata
    Arguments
        transform (torch.Transform)
        datafile (string) filename of a text file with an hdf5 file per row.
        preload (bool) load all data to memory, or just load from file on demand
    """
    def __init__(self, datafile, transform=None, preload=False):

        self.datafile = datafile
        self.transform = transform
        self.preload = preload

        # Find the input files
        self.h5_filenames = [line.strip() for line in open(datafile)]
        # count thenumber of elements per file.
        self.nsamples = 0
        self.file_range = ()
        for i in range(len(self.h5_filenames)):
            hf = h5py.File(self.h5_filenames[i], 'r')
            data = hf['Data']
            self.nsamples += data.shape[0]
            self.file_range += (self.nsamples,)


        self.images = None
        self.labels = None
        self.datalength = None
        if self.preload:
            for i in range(len(self.h5_filenames)):
                hf = h5py.File(self.h5_filenames[i], 'r')
                if self.images is None:
                    self.images = hf.get('Data')[()]
                    self.labels = hf.get('Labels')[()]
                    self.datalength = len(self.images)
                else:
                    data_ = hf.get('Data')
                    labels_ = hf.get('Labels')
                    # previous_rows = self.images.shape[0]
                    # self.images.resize((previous_rows+data_.shape[0],self.images.shape[1], self.images.shape[2]))
                    # self.images[previous_rows:previous_rows+data_.shape[0],:,:] = data_
                    # self.labels.resize((1,previous_rows + labels_.shape[1]))
                    # self.labels[:,previous_rows:previous_rows + labels_.shape[1], :, :] = labels_
                    self.images = np.concatenate((self.images, data_[()]), axis=0)
                    self.labels = np.concatenate((self.labels, labels_[()]), axis=1)


    def index_to_file_index(self, idx):
        file_id = 0
        index_in_file_id = 0
        range_starts = (0,)+ self.file_range

        for ind, v in enumerate(self.file_range):
            if idx < v:
                file_id = ind
                index_in_file_id = idx-range_starts[ind]
                break

        return file_id, index_in_file_id

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        """

        Arguments
            index (int) index position to return the data

        Returns
            image from the input index, and labels form metadata
        """

        if self.preload:
            l = self.labels[0, index] # The first column is the actual label
            B=np.zeros(len(self.images)) # create an array of zeros
            for i in range(len(self.file_range)-1):
                B[self.file_range[i]:self.file_range[i+1]]=self.file_range[i] # assign to B the number of frames of the next patient which needs to be added to the self.labels because all the patients were concatanated meaning if we use the original label index they start from 1 each time which isn't right
            current_image_index=self.labels[1, index]+B[index]
            previous_slice_image_index=self.labels[2, index]+B[index]
            img_slice_current = self.images[int(current_image_index-1), :, :]  # .astype(np.float32)
            img_slicebefore = self.images[int(previous_slice_image_index-1), :, :]  # Index of the slice before

        else:
            (file_id, index_in_file_id) = self.index_to_file_index(index)
            hf = h5py.File(self.h5_filenames[file_id], 'r')
            l = hf['Labels'][0,index_in_file_id]
            current_image_index = hf['Labels'][1,index_in_file_id]
            previous_slice_image_index = hf['Labels'][2,index_in_file_id]
            img_slice_current = hf['Data'][int(current_image_index),...]
            img_slicebefore = hf['Data'][int(previous_slice_image_index),...]  # Index of the slice before

        if self.transform is not None:
            img_slice_current = self.transform(np.float32(img_slice_current))
            img_slicebefore = self.transform(np.float32(img_slicebefore)) # need to change to float32 for the transforms
            img = np.concatenate((img_slicebefore, img_slice_current), axis=0)  # concatanate all these slices together
        return img, l
