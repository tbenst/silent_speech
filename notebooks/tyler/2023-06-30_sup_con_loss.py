##
import torch, torchmetrics, os, sys
import torch.nn.functional as F


# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from contrastive import supervised_contrastive_loss, supNCE_masks


##
L = 10
embeddings = torch.randn(L, 128)
labels = torch.randint(0, 10, (L,))
lb = labels.view(-1, 1)
mask = torch.eq(lb, lb.T)
C = torch.max(labels) + 1
L = labels.shape[0]
cardinality = torch.bincount(labels)
c = torch.where(cardinality > 1)[0][0]
c_idx = torch.where(labels == c)[0]
class_masks = torch.stack([labels == c for c in range(C)]) # C x L
c_mask = class_masks[c].unsqueeze(1) * class_masks[c].unsqueeze(0)

positives_mask = torch.einsum('cd,ce->cde', class_masks, class_masks)
assert torch.all(torch.eq(c_mask, positives_mask[c])), f"c_mask\n{c_mask}\npositives_mask\n{positives_mask[c_idx]}"
# negatives_mask = (~positives_mask).float()
# positives_mask = positives_mask.diagonal(dim1=1, dim2=2).fill_(0)
# positives_mask = positives_mask.float()
# assert positives_mask[0].sum() == len(z_idx) * (len(z_idx) - 1), positives_mask[0].sum()
##
def supervised_contrastive_loss(embeddings, labels, temperature=0.07, device="cpu"):
    """
    Compute supervised contrastive loss for a batch of embeddings. Skip classes with only one sample.
    
    Args:
        embeddings (torch.Tensor): [N x D]
        labels (torch.Tensor): [N]
        temperature (float, optional): Defaults to 0.07.
        device (str, optional): Defaults to "cpu".
    """
    N, D = embeddings.shape
    class_masks, positives_mask, negatives_mask = supNCE_masks(labels, device=device) # C x N x N
    # count number of positives for each class
    cardinality = torch.bincount(labels)
    classes = torch.where(cardinality > 1)[0] # Skip classes with only one sample
    C = classes.shape[0] + 1
    
    similarity_matrix = torchmetrics.functional.pairwise_cosine_similarity(
        embeddings, embeddings) / temperature
    similarity_matrix = torch.exp(similarity_matrix)
    # calculate per-class loss, dividing by the number of positives
    
    class_loss = torch.zeros(C, device=device)
    for i,c in enumerate(classes):
        nominator = positives_mask[c] * similarity_matrix
        nominator = torch.sum(nominator, dim=1)[class_masks[c]] # samples of proper class only
        denominator = negatives_mask[c] * similarity_matrix
        denominator = torch.sum(denominator, dim=1)[class_masks[c]]
        # sum over samples of proper class, divide by number of positives
        class_loss[i] = -torch.log(nominator / denominator).sum() / cardinality[c]
    return class_loss.sum() / N

x = F.normalize(torch.randn(L, 128))
supervised_contrastive_loss(x, labels, temperature=0.1)
##
