import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np, torchmetrics
import scipy.signal, logging
import scipy.io
from scipy.signal import iirnotch, lfilter
from typing import List


class KoLeoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(p=2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        x_flat = x.view(-1, x.size(-1))
        dots = torch.matmul(x_flat, x_flat.t())
        n = x_flat.shape[0]
        dots.view(-1)[::n+1].fill_(-1)
        _, I = torch.max(dots, dim=1)

        return I

    def forward(self, emg_latent, emg_parallel_latent, eps=1e-8):
        joint_embedding = torch.cat([emg_latent, emg_parallel_latent], dim=-1)
        joint_embedding = F.normalize(joint_embedding, p=2, dim=-1, eps=eps)
        
        I = self.pairwise_NNs_inner(joint_embedding)
        distances = self.pdist(joint_embedding.view(-1, joint_embedding.size(-1)), 
                               joint_embedding.view(-1, joint_embedding.size(-1))[I])
        
        loss = -torch.log(distances + eps).mean()
        return loss

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

def nobatch_cross_contrastive_loss(x,y, cos_sim=None, temperature=0.1, device='cpu'):
    """Compute cross contrastive loss between two sequences of embeddings.
    
    This diverges from the SimCLR paper in that we consider y_i to be a positive example of x_i, and vice versa,
    such that we have 1 positive examples, and 2L-2 negative examples for each x_i and y_i.
    
    Args:
        x: L x D
        y: L x D
        cos_sim (optional): L x L matrix of cosine similarities between x and y
        
    """
    assert x.shape[0] == y.shape[0], f"{x.shape=}, {y.shape=}"
    assert x.shape[1] == y.shape[1], f"{x.shape=}, {y.shape=}"
    L, D = x.shape
    
    positives_mask, denominator_mask = infoNCE_masks(L, device=device)
    if cos_sim is None:
        representations = torch.cat([x,y], dim=0)
        cos_sim = torchmetrics.functional.pairwise_cosine_similarity(representations,representations)
    similarity_matrix = torch.exp(cos_sim / temperature)
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

def supervised_contrastive_loss(embeddings, labels, phoneme_inventory, phoneme_weights, cos_sim=None, temperature=0.1, device="cpu"):
    """
    Compute supervised contrastive loss for a batch of embeddings. Skip classes with only one sample.
    
    Note: precomputing cosine similarities is not faster than computing them on the fly, likely due to cost of
    passing large tensor as argument.
    
    Args:
        embeddings (torch.Tensor): [N x D]
        labels (torch.Tensor): [N]
        cos_sim (torch.Tensor, optional): precomputed [N x N] matrix of cosine similarities between embeddings. Defaults to None.
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
    
    if cos_sim is None:
        cos_sim = torchmetrics.functional.pairwise_cosine_similarity(embeddings, embeddings)
    similarity_matrix = cos_sim / temperature
    similarity_matrix = torch.exp(similarity_matrix)
    # print(f"{similarity_matrix.shape}")
    # calculate per-class loss, dividing by the number of positives
    
    class_loss = torch.zeros(C, device=device)
    # logging.debug(f"{N=}, {D=}, {positives_mask.shape=}, {negatives_mask.shape=}, {class_masks.shape=}, {similarity_matrix.shape=}")
    logging.debug(f"{classes=}")
    for i,c in enumerate(classes):
        c = int(c)
        phoneme_label = phoneme_inventory[c]
        phoneme_weight = phoneme_weights.get(phoneme_label, 1.0)
        positives_mask, denominator_mask = supNCE_mask(labels, c, device=device)
        nominator = positives_mask * similarity_matrix
        # nominator = similarity_matrix[positives_mask]
        nominator = torch.sum(nominator, dim=1)[class_masks[c]] # samples of proper class only
        # print(f"{nominator=}")
        denominator = denominator_mask * similarity_matrix
        # denominator = similarity_matrix[denominator_mask]
        denominator = torch.sum(denominator, dim=1)[class_masks[c]]
        # print(f"{denominator_mask=}")
        # print(f"{denominator=}")
        # sum over samples of proper class, divide by number of positives
        class_loss[i] = (-torch.log(nominator / denominator).sum() / cardinality[c]) * phoneme_weight

    return class_loss.sum()

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature # https://github.com/HobbitLong/SupContrast/issues/106

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        features = F.normalize(features, dim=2)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            # Find the classes with more than one sample
            class_counts = labels.view(-1).bincount()
            classes_with_more_than_one_sample = (class_counts > 1)

            # Create a sample-wise mask indicating whether each sample belongs to a class with more than one member
            sample_mask = classes_with_more_than_one_sample[labels.view(-1)].float().to(device)

            # Modify the mask to only include samples from classes with more than one sample
            mask = torch.eq(labels, labels.T).float().to(device) * sample_mask.view(-1, 1) * sample_mask.view(1, -1)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(torch.clamp(exp_logits.sum(1, keepdim=True), min=1e-7))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss