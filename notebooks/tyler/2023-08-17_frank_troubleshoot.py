##
import scipy.io
sent = scipy.io.loadmat('/data/data/T12_data/sentences/t12.2022.07.27_sentences.mat')
comp = scipy.io.loadmat('/data/data/T12_data/competitionData/train/t12.2022.07.27.mat')

import matplotlib.pyplot as plt
import numpy as np

loopIdx = np.arange(sent['goTrialEpochs'][187,0], sent['goTrialEpochs'][187,1]).astype(np.int32)
spikePow = sent['spikePow'][loopIdx,:]
spikePow_comp = comp['spikePow'][0,187]

plt.figure()
plt.plot(spikePow[:,0])
plt.plot(spikePow_comp[:,0],'--')
plt.show()

print(np.mean(spikePow[:,0]))
print(np.mean(spikePow_comp[:,0]))