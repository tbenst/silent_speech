##
import torch, torchmetrics, sys, os
import torch.nn.functional as F
import numpy as np

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from contrastive import cross_contrastive_loss, var_length_cross_contrastive_loss, \
    nobatch_cross_contrastive_loss, supervised_contrastive_loss

B,D = 3,5

bz = 4
k = 0.1
for L in range(2,5):
    L_dub = 2*L
    # skip every other class
    my_class = torch.arange(L, dtype=torch.long).repeat(2) * 2
    # https://github.com/HobbitLong/SupContrast/issues/106#issue-1192097371
    # print(f"Testing (B,L,D) = ({bz},{L},{D})")
    theory = torch.log(torch.tensor(2*L-1))
    x = F.normalize(torch.ones(bz,L,L), dim=-1)
    x_dub = F.normalize(torch.ones(L_dub,L_dub), dim=-1)
    cc = cross_contrastive_loss(x,x,temperature=k)
    # print("NO BATCH")
    nbcc = nobatch_cross_contrastive_loss(x[0],x[0],temperature=k)
    # print("SUPERVISED")
    scc = supervised_contrastive_loss(x_dub,my_class,temperature=k)
    # TODO: need to fix cross_contrastive_loss to match theory...
    assert torch.isclose(cc, theory), (cc, theory)
    assert torch.isclose(nbcc, theory), (nbcc, theory)
    assert torch.isclose(scc, theory), (scc, theory)
    