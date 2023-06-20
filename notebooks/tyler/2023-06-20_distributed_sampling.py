##

import numpy as np, torch, torch.distributed as dist, sys, os
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from dataloaders import StratifiedBatchSampler


class DistributedStratifiedBatchSampler(StratifiedBatchSampler):
    """"Given the class of each example, sample batches without replacement
    with desired proportions of each class.
    
    If we run out of examples of a given class, we stop yielding batches.
    """
    def __init__(self, classes:np.ndarray, class_proportion:np.ndarray,
                 batch_size:int, shuffle:bool=True, drop_last:bool=False, seed:int=61923):        
        self.num_replicas = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        assert batch_size % self.num_replicas == 0, "Batch size must be divisible by number of GPUs"
        internal_bz = batch_size // self.num_replicas
        super().__init__(classes, class_proportion, internal_bz, shuffle, drop_last)
        mod = self.num_batches % self.num_replicas
        if mod != 0:
            self.num_batches -= mod
        
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        if "RANK" not in os.environ:
            print("WARNING: RANK not in environment, setting to 0")
        else:
            print(f"HURRAY! Got rank {os.environ['RANK']}")

        self.rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
        self.epoch = 0
        self.seed = seed
        
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.class_indices = [x[torch.randperm(len(x), generator=g).tolist()]
                                    for x in self.class_indices]
        for batch in range(self.rank,self.num_batches,self.num_replicas):
            batch_indices = []
            for i in range(self.class_n_per_batch.shape[0]):
                s = batch * self.class_n_per_batch[i]
                e = (batch+1) * self.class_n_per_batch[i]
                idxs = self.class_indices[i][s:e]
                batch_indices.extend(idxs)
            batch_indices = [int(x) for x in batch_indices]
            # not needed for our purposes as model doesn't care about order
            # if self.shuffle:
            #     batch_indices = np.random.permutation(batch_indices)
            yield batch_indices
            
    def __len__(self):
        return int(np.floor(self.num_batches / self.num_replicas))
    
    
def main():
    x = np.arange(17)
    classes = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
    sampler = DistributedStratifiedBatchSampler(classes, np.array([0.5, 0.5]), 4, shuffle=False)
    for i in sampler:
        print(f"({i}) Rank {os.environ['RANK']}: {x[i]=} {classes[i]=}")
    print(f"{len(sampler)=}, {sampler.num_batches=}, {sampler.num_replicas=}, {sampler.rank=}")
    
    
"""torchrun --standalone --nnodes=1 --nproc_per_node=2 2023-06-20_distributed_sampling.py"""
if __name__ == "__main__":
    # dist.init_process_group()
    print(f"Rank {os.environ['RANK']} of {os.environ['WORLD_SIZE']}")
    main()
##
