from ahaseg import LVseg
import numpy as np
import matplotlib.pyplot as plt

pred_mask = np.load(r'testmask.npy')
LVSeg_label = LVseg(pred_mask==3, pred_mask==2, pred_mask==1, nseg=6)
#plt.imshow(LVSeg_label)
#plt.show()


from ahaseg import AHA17

temp1 = np.load(r'example.npz')
T1map = temp1['raw256']
pred_mask = temp1['label_predict']
tempB = dict()
tempB['Qmap'] = T1map
tempB['RVb_mask'] = (pred_mask ==1)
tempB['LVw_mask'] = (pred_mask ==2)
tempB['LVb_mask'] = (pred_mask ==3)


AHAdt = AHA17(B=tempB,M=tempB,A=tempB,figname=None)
print(AHAdt['mean17'])

AHAdt = AHA17(B=tempB,M=tempB,A=tempB,figname='test.jpg')
print(AHAdt['mean17'])