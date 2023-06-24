##
import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim = 256,
    num_quantizers = 4,      # specify number of quantizers
    codebook_size = 1024,    # codebook size
)

x = torch.randn(1, 24, 256)

quantized, indices, commit_loss = residual_vq(x)
print(f"quantized - {quantized.shape}\nindices - {indices.shape}\ncommit_loss - {commit_loss.shape}")
##
# (1, 24, 256), (1, 24, 4), (1, 4)
# (batch, seq, dim), (batch, seq, quantizer), (batch, quantizer)

# if you need all the codes across the quantization layers, just pass return_all_codes = True

quantized, indices, commit_loss, all_codes = residual_vq(x, return_all_codes = True)
print(f"quantized - {quantized.shape}\nindices - {indices.shape}\ncommit_loss - {commit_loss.shape}\nall_codes - {all_codes.shape}")
# *_, (4, 1, 24, 256)
# all_codes - (quantizer, batch, seq, dim)
##
