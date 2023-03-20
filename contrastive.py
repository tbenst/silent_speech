import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.signal
import scipy.io
from scipy.signal import iirnotch, lfilter

class ContrastiveLoss(nn.Module):
    def __init__(self, device = 'cpu', temperature=0.5):
        super().__init__()
        self.device     = device
        self.register_buffer("temperature", torch.tensor(temperature))
        #self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        
        batch_size     = emb_i.shape[0]
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device= self.device)).float()

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
    
    

class DataAugmenter:
    def __init__(self, augmentations, total_points=None, window_size=None, sfreq= 800, bw=5, 
                 num_channels=8, num_augmentations = 2, temporal_len=3000,):
        
        self.available_augmentations = augmentations
        self.TEMPORAL_DIM = 0
        self.CHANNEL_DIM = 1
        self.NUM_AUGMENTATIONS = num_augmentations
        self.NUM_CHANNELS = num_channels
        self.TEMPORAL_LEN = temporal_len
        self.SFREQ = sfreq
        self.BW = bw # band width (?) see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
        
        
    def transform(self, x):
        '''Given data sample x, apply random augmentations.'''

        x_aug = np.copy(x)
       
        curr_augmentations = self.get_augmentation_set()
        x_aug             = self.apply_augmentations(x_aug, curr_augmentations)

        return x, torch.tensor(x_aug)

    

    def get_augmentation_set(self):
        """ Generates a set of augmentations for each channel (up to self.NUM_AUGMENTATIONS) many.
            Returns:
                
                augmentation_set (list of dict) - elements are augmentation types and associated values;
                                                  one dictionary for each channel  
        """
        augmentation_set = [] 
        
        for j in range(self.NUM_CHANNELS):
            augmentation_set.append(dict())
            selected_augmentations = np.random.choice(list(self.available_augmentations.keys()), self.NUM_AUGMENTATIONS) # see https://pynative.com/python-random-sample/#:~:text=Python's%20random%20module%20provides%20random,it%20random%20sampling%20without%20replacement.
            for _, curr_augmentation in enumerate(selected_augmentations):
                curr_augmentation_val = None

                if curr_augmentation in ['amplitude_scale', 'DC_shift', 'additive_Gaussian_noise', 'band-stop_filter']: # augmentation that requires float val
                    curr_augmentation_val = np.random.uniform(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range

                elif curr_augmentation in ['time_shift', 'zero-masking']: # augmentation that requires int val
                    curr_augmentation_val = np.random.randint(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/3996904/generate-random-integers-between-0-and-9
                    if curr_augmentation == 'zero-masking':
                        curr_augmentation_val = [curr_augmentation_val, np.random.randint(0, self.TEMPORAL_LEN-1)]

                else:
                    raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for value sampling")

                augmentation_set[j][curr_augmentation] = curr_augmentation_val
            
        return augmentation_set
    
    
    def apply_augmentations(self, x, augmentations):
        """ Applies augmentations to channels in EMG sample <x>. Inputs are:
        
            x (2D float) - time x channels numpy array
            augmentations (list of dict) - entries contain augmentation types and strengths to
                                            be applied to a given channel
        
        see Section 2.2 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        
        assert len(augmentations) == self.NUM_CHANNELS
        for j, curr_augmentation_set in enumerate(augmentations):
            for _, curr_augmentation in enumerate(list(curr_augmentation_set.keys())):
                curr_augmentation_val = curr_augmentation_set[curr_augmentation]

                if curr_augmentation == 'amplitude_scale':
                    x[:,j] = curr_augmentation_val * x[:,j]
                    
                elif curr_augmentation == 'DC_shift':
                    x[:,j] = x[:,j] + curr_augmentation_val
                    
                elif curr_augmentation == 'additive_Gaussian_noise':
                    x[:,j] = x[:,j] + np.random.normal(0, curr_augmentation_val, x[:,j].shape)# see https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python and https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
                    
                elif curr_augmentation == 'band-stop_filter':
                    """
                    see:
                        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
                        https://www.programcreek.com/python/example/115815/scipy.signal.iirnotch
                        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
                    """

                    b, a = iirnotch(curr_augmentation_val, curr_augmentation_val/self.BW, self.SFREQ)
                    x[:,j] = lfilter(b, a, x[:,j])

                elif curr_augmentation == 'time_shift':
                    if curr_augmentation_val != 0:
                        new_signal = np.zeros(x[:,j].shape)

                        if curr_augmentation_val < 0:
                            new_signal[:curr_augmentation_val] = x[np.abs(curr_augmentation_val):,j]
                            new_signal[curr_augmentation_val:] = x[:np.abs(curr_augmentation_val),j]
                        else:
                            new_signal[:curr_augmentation_val] = x[-curr_augmentation_val:,j]
                            new_signal[curr_augmentation_val:] = x[:-curr_augmentation_val,j]

                        x[:,j] = new_signal

                elif curr_augmentation == 'zero-masking': 
                    x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j] = 0.
                else:
                    raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for application")
        
        return x