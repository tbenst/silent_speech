##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch, torchmetrics, sys
import torch.nn.functional as F
import numpy as np, timeit
from magneto import bench

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from contrastive import cross_contrastive_loss, var_length_cross_contrastive_loss, \
    nobatch_cross_contrastive_loss, supervised_contrastive_loss, infoNCE_masks, supNCE_mask

##
L = 4096
x = torch.rand(L,256)
y = torch.rand(L,256)
cos_sim = torchmetrics.functional.pairwise_cosine_similarity(x,y)
cos_sim.shape


c1 = bench(nobatch_cross_contrastive_loss)(x, y) # 0.281 (CPU)
representations = torch.cat([x,y], dim=0)
cos_sim = torchmetrics.functional.pairwise_cosine_similarity(representations,representations)
c2 = bench(nobatch_cross_contrastive_loss)(x, y, cos_sim=cos_sim) # 0.206 (CPU)
assert torch.allclose(c1,c2)

##
x_class = torch.arange(L//4, dtype=torch.long).repeat(4)
s1 = bench(supervised_contrastive_loss, 1)(x, x_class) # 52.23s (CPU)
cos_sim = torchmetrics.functional.pairwise_cosine_similarity(x,x)
s2 = bench(supervised_contrastive_loss, 1)(x, x_class, cos_sim=cos_sim) # 56.02s (CPU)
assert torch.allclose(s1,s2)
##
# BEGIN: 8f7a3b5d4e6c
!pip install line_profiler

%load_ext line_profiler

%lprun -f supervised_contrastive_loss supervised_contrastive_loss(x, x_class)
# END: 8f7a3b5d4e6c
supervised_contrastive_loss(x, x_class)

##
L = 4096
similarity_matrix = torch.rand(L,L)
labels = torch.arange(L//4, dtype=torch.long).repeat(4)
positives_mask, denominator_mask = supNCE_mask(labels, 1)
def mult_index(positives_mask, denominator_mask):
    return positives_mask * similarity_matrix
    
def normal_index(positives_mask, denominator_mask):
    return similarity_matrix[positives_mask]

def gather_index(positives_mask, denominator_mask):
    positives_indices = torch.nonzero(positives_mask).t()
    return torch.gather(similarity_matrix, 1, positives_indices)
##
print(bench(mult_index)(positives_mask, denominator_mask).shape) # 0.02 se
print(bench(normal_index)(positives_mask, denominator_mask)) # 0.0015 se
print(bench(gather_index)(positives_mask, denominator_mask)) # 0.0014 se
##
similarity_matrix.shape, positives_mask.shape

##
gather_index(positives_mask, denominator_mask).shape

##
labels = torch.arange(L//4, dtype=torch.long).repeat(4)
L = len(labels)
class_masks = {}
for c in labels:
    c = int(c)
    class_masks[c] = labels == c
my_class = 2
class_mask = labels == my_class
positives_mask = torch.einsum('d,e->de', class_mask, class_mask)
positives_mask.diagonal().fill_(False) # ignore self-similarity
denominator_mask = ~torch.eye(L, L, dtype=bool)

# get the indices where positives_mask is True
positives_indices = torch.nonzero(positives_mask).t()

# Use gather to get the elements at these indices
nominator = torch.gather(similarity_matrix, 1, positives_indices)
nominator = torch.sum(nominator, dim=1)[class_masks[c]] # samples of proper class only

##
import torch, torchmetrics
# Analyze why the following script is slow & uses a lot of memory.

# ```python
L = 4096
# x = torch.rand(L,256).cuda()
x = torch.rand(L,256)
nclass = 32
n = (L // nclass)
x_class = torch.arange(L// n, dtype=torch.long).repeat(n)


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
    positives_mask = torch.einsum('cd,ce->cde', class_masks, class_masks)
    positives_mask.diagonal(dim1=1, dim2=2).fill_(False) # ignore self-similarity
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

def supervised_contrastive_loss1(embeddings, labels, cos_sim=None, temperature=0.1, device="cpu"):
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

supervised_contrastive_loss1(x, x_class)

##
from magneto import bench

L = 4096
x = torch.rand(L, 256)
nclass = 32
n = (L // nclass)
x_class = torch.arange(n, dtype=torch.long).repeat(nclass)

def supervised_contrastive_loss_optimized_batched(embeddings, labels, batch_size=512, temperature=0.1, device="cpu"):
    """
    Compute supervised contrastive loss for a batch of embeddings. Skip classes with only one sample.

    Args:
        embeddings (torch.Tensor): [N x D]
        labels (torch.Tensor): [N]
        batch_size (int, optional): Defaults to 512.
        temperature (float, optional): Defaults to 0.07.
        device (str, optional): Defaults to "cpu".
    """
    N, D = embeddings.shape
    assert N == len(labels), f"Number of embeddings ({N}) and labels ({len(labels)}) must match"

    # count number of positives for each class
    cardinality = torch.bincount(labels) - 1  # number of comparisons per class
    classes = torch.where(cardinality > 0)[0]  # Skip classes with only one sample
    C = classes.shape[0] + 1

    class_loss = torch.zeros(C, device=device)

    # Loop over classes
    for i, c in enumerate(classes):
        c = int(c)

        # Compute indices for the current class
        indices = torch.where(labels == c)[0]

        nominator = torch.zeros(len(indices), device=device)
        denominator = torch.zeros(len(indices), device=device)

        # Loop over batches
        for j in range(0, N, batch_size):
            batch_indices = torch.arange(j, min(j + batch_size, N), device=device)
            # Compute cosine similarity matrix for the current batch
            cos_sim = F.cosine_similarity(embeddings[indices].unsqueeze(1), embeddings[batch_indices].unsqueeze(0), dim=-1)
            similarity_matrix = torch.exp(cos_sim / temperature)

            # Compute mask for the current batch
            mask = (batch_indices[None, :] == indices[:, None])

            # Compute nominator and denominator for the current batch
            nominator += torch.sum(similarity_matrix, dim=1) - torch.sum(similarity_matrix * mask, dim=1)
            denominator += torch.sum(similarity_matrix, dim=1) - 1  # subtract self-similarity

        # Subtract nominator from denominator
        denominator -= nominator

        # Compute class loss
        class_loss[i] = -torch.log(nominator / denominator).sum() / cardinality[c]

    return class_loss.sum()

bench(supervised_contrastive_loss_optimized_batched, 1)(x, x_class) # 14.56s
##
# original
def supervised_contrastive_loss(embeddings, labels, cos_sim=None, temperature=0.1, device="cpu"):
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
    for i,c in enumerate(classes):
        c = int(c)
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
        class_loss[i] = -torch.log(nominator / denominator).sum() / cardinality[c]
    return class_loss.sum()

bench(supervised_contrastive_loss, 1)(x, x_class) # 1.54s

##
import numpy as np
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    
    https://github.com/HobbitLong/SupContrast
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature # https://github.com/HobbitLong/SupContrast/issues/106

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

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
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
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(torch.clamp(exp_logits.sum(1, keepdim=True), min=1e-7))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
the_answer = np.log(15).astype(np.float32)
print("the answer we want", the_answer)                 
                                                        
featTensor = torch.ones((8, 2, 10))                     
featTensor = torch.nn.functional.normalize(featTensor, dim=-1)
                                                        
criterion = SupConLoss(                                 
        temperature=1.0,                                
        base_temperature=1.0)                           
                                                        
loss = criterion(featTensor)
assert torch.isclose(torch.tensor(the_answer), loss), loss
##
bench(criterion, 1)(x[:, None], x_class) # 1.54s
##
L = 128
k = 0.1
L_dub = 2*L
my_class = torch.arange(L, dtype=torch.long).repeat(2) * 2
theory = torch.log(torch.tensor(2*L-1))
x_dub = F.normalize(torch.ones(L_dub,L_dub), dim=-1)
sc_corrected = criterion(x_dub[:, None], my_class)
sc_orig = supervised_contrastive_loss(x_dub, my_class)
assert torch.isclose(sc_orig / (L*2), theory, atol=1e-3), (sc_orig, theory)
assert torch.isclose(sc_corrected, theory, atol=1e-3), (sc_corrected, theory)
sc_corrected
##
class ModSupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(ModSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature

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

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
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
        log_prob = logits - torch.log(torch.clamp(exp_logits.sum(1, keepdim=True), min=1e-7))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
crit = ModSupConLoss()

loss = crit(featTensor)
assert torch.isclose(torch.tensor(the_answer), loss), loss
##
L = 4096
x = torch.rand(L, 256)
nclass = 32
n = (L // nclass)
x_class = torch.arange(n, dtype=torch.long).repeat(nclass) 
ModSupConLoss()(x[:,None], x_class)
x_class[0] = x_class.max() + 1

print(bench(ModSupConLoss(contrast_mode='one'), 1)(x[:,None], x_class))
print(bench(SupConLoss(), 1)(x[:,None], x_class))
print(bench(supervised_contrastive_loss, 1)(x, x_class))
##
