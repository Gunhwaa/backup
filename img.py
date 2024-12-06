import cv2
from glob import glob

root = '/home/dgkim/workspace/tr3d/lwir'
for im in glob('/home/dgkim/workspace/tr3d/data/sunrgbd/sunrgbd_trainval/lwir/*'):
    
    lwir = cv2.imread(im)
    lwir_reshape = cv2.resize(lwir, (730, 530))

    cv2.imwrite(f"/home/dgkim/workspace/tr3d/lwir/{im.split('/')[-1]}", lwir_reshape)