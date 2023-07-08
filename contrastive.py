import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np, torchmetrics
import scipy.signal, logging
import scipy.io
from scipy.signal import iirnotch, lfilter
from typing import List

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
    
def pairwise_cos_sim(x, y):
    """Batched cosine similarity matrix.
    
    Args:
        x: B x N x D
        y: B x M x D
        returns: B x N x M matrix of cosine similarities between all pairs
    """
    x = F.normalize(x, dim=2)
    y = F.normalize(y, dim=2)
    return torch.bmm(x,y.permute(0,2,1))

def infoNCE_masks(L, device='cpu'):
    "Return nominator and denominator masks for infoNCE loss."
    positives_mask = torch.zeros(L*2, L*2, dtype=bool, device=device)
    torch.diagonal(positives_mask, offset=L).fill_(True) # x_i is a positive example of y_i
    torch.diagonal(positives_mask, offset=-L).fill_(True) # y_i is a positive example of x_i
    
    # denominator_mask = (~torch.eye(L*2, L*2, dtype=bool, device=device)) # over everything (original formulation)
    # denominator_mask = (~torch.eye(L*2, L*2, dtype=bool, device=device))
    # return positives_mask, denominator_mask
    negatives_mask = positives_mask.clone()
    torch.diagonal(negatives_mask).fill_(True) # ignore self-similarity
    negatives_mask = (~negatives_mask).float()
    return positives_mask, negatives_mask


def supNCE_masks(labels:torch.Tensor, device='cpu'):
    """Return nominator and denominator masks for supervised contrastive loss.
    
    Args:
        labels: L tensor of integer labels
    
    Returns:
        positives_mask: C x L x L mask of positive examples
        denominator_mask: C x L x L mask for denominator
        
    where C is the number of classes, and L is the number of samples.
    Note: memory usage is approximately O(2*C*L^2), so this is not suitable for large datasets.
    """
    L = len(labels)
    C = torch.max(labels) + 1
    # for each class, return mask where i,j is True if i and j are in the same class
    class_masks = torch.stack([labels == c for c in range(C)]) # C x L
    logging.debug(f"{class_masks.shape=}")
    positives_mask = torch.einsum('cd,ce->cde', class_masks, class_masks)
    positives_mask.diagonal(dim1=1, dim2=2).fill_(False) # ignore self-similarity
    logging.debug(f"{positives_mask.shape=}")
    denominator_mask = torch.ones(C, L, L, dtype=bool, device=device)
    # ignore self-similarity
    torch.diagonal(positives_mask, dim1=1, dim2=2).fill_(False)
    return class_masks, positives_mask, denominator_mask

def supNCE_mask(labels:torch.Tensor, my_class:int, device='cpu'):
    "Return nominator and denominator masks for one class of supervised contrastive loss."
    L = len(labels)
    class_mask = labels == my_class
    positives_mask = torch.einsum('d,e->de', class_mask, class_mask)
    positives_mask.diagonal().fill_(False) # ignore self-similarity
    denominator_mask = ~torch.eye(L, L, dtype=bool, device=device)
    return positives_mask, denominator_mask
    

def cross_contrastive_loss(x,y, temperature=0.1, device='cpu'):
    """Compute cross contrastive loss between two batches of embeddings.
    
    This diverges from the SimCLR paper in that we consider y_i to be a positive example of x_i, and vice versa,
    such that we have 1 positive examples, and 2L-2 negative examples for each x_i and y_i.
    
    Args:
        x: B x L x D
        y: B x L x D
        
    """
    B, L, D = x.shape
    representations = torch.cat([x,y], dim=1)
    
    positives_mask, denominator_mask = infoNCE_masks(L, device=device)
    similarity_matrix = torch.exp(pairwise_cos_sim(representations,representations) / temperature)
    nominator = similarity_matrix[positives_mask.expand(B,-1,-1)] # now B*L*2
    assert nominator.shape == (B*L*2,), f"{nominator.shape=}, {B*L*2=}"

    denominator = denominator_mask * similarity_matrix
    denominator = torch.sum(denominator, dim=1).reshape(B*L*2)
    loss_partial = -torch.log(nominator / denominator)
    loss = torch.mean(loss_partial)
    return loss

def nobatch_cross_contrastive_loss(x,y, temperature=0.1, device='cpu'):
    """Compute cross contrastive loss between two sequences of embeddings.
    
    This diverges from the SimCLR paper in that we consider y_i to be a positive example of x_i, and vice versa,
    such that we have 1 positive examples, and 2L-2 negative examples for each x_i and y_i.
    
    Args:
        x: L x D
        y: L x D
        
    """
    L, D = x.shape
    representations = torch.cat([x,y], dim=0)
    
    positives_mask, denominator_mask = infoNCE_masks(L, device=device)
    
    similarity_matrix = torch.exp(torchmetrics.functional.pairwise_cosine_similarity(
        representations,representations) / temperature)
    # print(f"{similarity_matrix.shape}")
    # TODO pretty sure can do exp just once, double check & change
    nominator = similarity_matrix[positives_mask] # now L*2
    # print(f"{nominator=}")
    denominator = denominator_mask * similarity_matrix
    denominator = torch.sum(denominator, dim=1).reshape(L*2)
    # print(f"{denominator_mask=}")
    # print(f"{denominator=}")
    loss_partial = -torch.log(nominator / denominator)
    loss = torch.mean(loss_partial) # average loss per sample
    return loss

def var_length_cross_contrastive_loss(x:List[torch.Tensor], y:List[torch.Tensor],
                                      temperature=0.1, device="cpu"):
    """
    Compute cross contrastive loss between two batches of embeddings,
    where the length of the sequences in each batch may vary.

    Args:
        x (List[torch.Tensor]): [N x D, M x D, ...]
        y (List[torch.Tensor]): [N x D, M x D, ...]
        k (float, optional): temperature. Defaults to 0.1.
        device (str, optional): Defaults to "cpu".
    """
    loss = 0.
    for i in range(len(x)):
        loss += nobatch_cross_contrastive_loss(x[i], y[i], temperature=temperature, device=device)
    return loss / len(x)

def supervised_contrastive_loss(embeddings, labels, temperature=0.1, device="cpu"):
    """
    Compute supervised contrastive loss for a batch of embeddings. Skip classes with only one sample.
    
    Args:
        embeddings (torch.Tensor): [N x D]
        labels (torch.Tensor): [N]
        temperature (float, optional): Defaults to 0.07.
        device (str, optional): Defaults to "cpu".
    """
    N, D = embeddings.shape
    assert N == len(labels), f"Number of embeddings ({N}) and labels ({len(labels)}) must match"

    # count number of positives for each class
    cardinality = torch.bincount(labels) - 1 # number of comparisons per class
    classes = torch.where(cardinality > 0)[0] # Skip classes with only one sample
    C = classes.shape[0] + 1
    class_masks = {}
    for c in classes:
        c = int(c)
        class_masks[c] = labels == c
    
    similarity_matrix = torchmetrics.functional.pairwise_cosine_similarity(
        embeddings, embeddings) / temperature
    similarity_matrix = torch.exp(similarity_matrix)
    # print(f"{similarity_matrix.shape}")
    # calculate per-class loss, dividing by the number of positives
    
    class_loss = torch.zeros(C, device=device)
    # logging.debug(f"{N=}, {D=}, {positives_mask.shape=}, {negatives_mask.shape=}, {class_masks.shape=}, {similarity_matrix.shape=}")
    logging.debug(f"{classes=}")
    for i,c in enumerate(classes):
        c = int(c)
        positives_mask, denominator_mask = supNCE_mask(labels, c, device=device)
        nominator = positives_mask * similarity_matrix
        nominator = torch.sum(nominator, dim=1)[class_masks[c]] # samples of proper class only
        # print(f"{nominator=}")
        denominator = denominator_mask * similarity_matrix
        denominator = torch.sum(denominator, dim=1)[class_masks[c]]
        # print(f"{denominator_mask=}")
        # print(f"{denominator=}")
        # sum over samples of proper class, divide by number of positives
        class_loss[i] = -torch.log(nominator / denominator).sum() / cardinality[c]
    return class_loss.sum()
