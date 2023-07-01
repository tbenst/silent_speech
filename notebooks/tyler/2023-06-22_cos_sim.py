##

import torch, torchmetrics
import torch.nn.functional as F

x = torch.rand(2,3)
y = torch.rand(4,3)
cos_sim = torchmetrics.functional.pairwise_cosine_similarity(x,y)
cos_sim_2 = torchmetrics.functional.pairwise_cosine_similarity(x,y)

f_cos_sim0 = F.cosine_similarity(x[0], y[0],dim=0)
f_cos_sim0_2 = F.cosine_similarity(x[0], y[0],dim=0)
f_cos_sim1 = F.cosine_similarity(x[1], y[1],dim=0)
f_cos_sim1_2 = F.cosine_similarity(x[1], y[1],dim=0)
assert torch.isclose(cos_sim[0,0], f_cos_sim0), f'{cos_sim[0,0]}, {cos_sim_2[0,0]} != {f_cos_sim0}, {f_cos_sim0_2}'
assert torch.isclose(cos_sim[1,1], f_cos_sim1), f'{cos_sim[1,1]}, {cos_sim_2[1,1]} != {f_cos_sim1}, {f_cos_sim1_2}'
##
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

def cross_contrastive_loss(x,y, k=0.1, device='cpu'):
    """Compute cross contrastive loss between two batches of embeddings.
    
    This diverges from the SimCLR paper in that we consider y_i to be a positive example of x_i, and vice versa,
    such that we have 1 positive examples, and 2L-2 negative examples for each x_i and y_i.
    
    Args:
        x: B x L x D
        y: B x L x D
        
    """
    B, L, D = x.shape
    representations = torch.cat([x,y], dim=1)
    
    positives_mask = torch.zeros(L*2, L*2, dtype=bool, device=device)
    torch.diagonal(positives_mask, offset=L).fill_(True) # x_i is a positive example of y_i
    torch.diagonal(positives_mask, offset=-L).fill_(True) # y_i is a positive example of x_i
    
    negatives_mask = positives_mask.clone()
    torch.diagonal(negatives_mask).fill_(True) # ignore self-similarity
    negatives_mask = (~negatives_mask).float()
    
    similarity_matrix = pairwise_cos_sim(representations,representations) / k
    positives = similarity_matrix[positives_mask.expand(B,-1,-1)] # now B*L*2
    nominator = torch.exp(positives)
    denominator = negatives_mask * torch.exp(similarity_matrix)
    denominator = torch.sum(denominator, dim=1).reshape(B*L*2)
    loss_partial = -torch.log(nominator / denominator)
    loss = torch.mean(loss_partial)
    return loss
    
# B,L,D = 2,3,4
B,L,D = 3,4,5
x = torch.randn(B,L,D)
y = torch.randn(B,L,D)
cross_contrastive_loss(x,y)

bz = 4
k = 0.1
for L in range(2,10):
    num_dist = (2*L-2)
    
    # best possible score
    x = torch.eye(L,L).expand(bz,-1,-1)
    k_inv = torch.tensor(1/k)
    optimal = -torch.log(torch.exp(k_inv)/num_dist)
    cc = cross_contrastive_loss(x,x,k=k)
    assert torch.isclose(cc, optimal), (cc, optimal)
    
    # score if all are the same
    x = torch.zeros(bz,L,L)
    all_same = -torch.log(torch.exp(k_inv)/(num_dist*torch.exp(k_inv)))
    cc = cross_contrastive_loss(x,x,k=k)
    assert torch.isclose(cc, all_same), (cc, all_same)
    
    # score if all x are the same, all y are the same, but x != y
    x = torch.ones(bz,L,L)
    x[:,:,:int(L/2)] = 0
    y = torch.ones(bz,L,L)
    xy_sim = F.cosine_similarity(x[0,0][None],y[0,0][None])
    # half distractors are the same, half are different
    denominator = num_dist/2*torch.exp(k_inv) + num_dist/2*torch.exp(xy_sim/k)
    xy_diff = -torch.log(torch.exp(xy_sim/k)/denominator)
    cc = cross_contrastive_loss(x,y,k=k)
    assert torch.isclose(cc, xy_diff), (cc, xy_diff)
    
    ## CCL for random x,y
    x = torch.randn(bz,L,L)
    y = torch.randn(bz,L,L)
    random_xy = cross_contrastive_loss(x,y,k=k)
    
    print(f"{L}: {optimal=}, {all_same=}, {xy_diff=}, {random_xy=}")
    
    
    

##
x = torch.rand(2,3,4)
y = torch.rand(2,5,4)
cos_sim = pairwise_cos_sim(x,y)

assert torch.isclose(
    torchmetrics.functional.pairwise_cosine_similarity(x[0],y[0])[1,1],
    F.cosine_similarity(x[0,1], y[0,1],dim=0)
)

a = torchmetrics.functional.pairwise_cosine_similarity(x[0],y[0])
b = cos_sim[0]
assert torch.all(torch.isclose(a,b)), f"\n{a}\n{b}"
assert torch.all(torch.isclose(x / torch.norm(x, p=2, dim=2).unsqueeze(2), F.normalize(x, dim=2)))
##
x = torch.rand(2,5,4)
y = pairwise_cos_sim(x,x)[0]
torch.diag(y,2), torch.diag(y,-2)
##
length = 2
positives_mask = torch.zeros(length*2, length*2, dtype=bool)
torch.diagonal(positives_mask, offset=length).fill_(True) # x_i is a positive example of y_i
torch.diagonal(positives_mask, offset=-length).fill_(True) # y_i is a positive example of x_i

negatives_mask = positives_mask.clone()
torch.diagonal(negatives_mask).fill_(True)
negatives_mask = (~negatives_mask).float()
x = torch.arange(length**2*4).reshape(length*2,length*2).float()
assert torch.equal(x[positives_mask], torch.tensor([ 2,  7,  8, 13]))

x = torch.randn(3,2*length,4)
sim_mat = pairwise_cos_sim(x,x)
assert (sim_mat * negatives_mask).sum(dim=1).shape == torch.Size([3, 4])
torch.equal(positives_mask.expand(3,-1,-1)[0],positives_mask)
torch.equal(positives_mask.expand(3,-1,-1)[1],positives_mask)
sim_mat[positives_mask.expand(3,-1,-1)]
##
