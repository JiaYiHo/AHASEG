from ahaseg import LVseg
import numpy as np
import matplotlib.pyplot as plt

pred_mask = np.load(r'testmask.npy')
LVSeg_label = LVseg(pred_mask==3, pred_mask==2, pred_mask==1, nseg=6)
plt.imshow(LVSeg_label)
plt.show()