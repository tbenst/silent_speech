##
import numpy as np

npz = np.load("/scratch/users/tbenst/2023-08-01T06:54:28.359594_gaddy/SpeechOrEMGToText-epoch=199-val/top100_50beams_thresh75_lmweight2.0_noLM.npz",
    allow_pickle=True)

npz['predictions'][:3]