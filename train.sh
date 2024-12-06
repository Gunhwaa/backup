
# # fog baseline super-vison
# CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=10 ./tools/dist_train.sh configs/tr3d/fog/super/fog_original_tr3d-ff_sunrgbd-10class.py 2

# # fog baselune semi-super-vison ~~~ only test
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 PORT=5524 python tools/test.py configs/tr3d/fog/semi/fog_original_tr3d-ff_sunrgbd-10class.py work_dirs/tr3d-ff_sunrgbd-3d-10class/latest.pth --eval mAP
## fog ~ use noise
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 PORT=5524 python tools/test.py configs/tr3d/fog/semi/fog_noise_original_tr3d-ff_sunrgbd-10class.py work_dirs/tr3d-ff_sunrgbd-3d-10class/latest.pth --eval mAP



# Normal MS 
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 PORT=5524 python tools/test.py configs/tr3d/ms/fog/super/fog_noise_ms_tr3d-ff_sunrgbd-10class_sum.py work_dirs/fog_ms_tr3d-ff_sunrgbd-10class/latest.pth --eval mAP
# CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=10 PORT=5523 ./tools/dist_train.sh configs/tr3d/ms/normal/ms_tr3d-ff_sunrgbd-10class.py 2 

# fog MS super
# CUDA_VISIBLE_DEVICES=6,7 PORT=5522 OMP_NUM_THREADS=10 ./tools/dist_train.sh configs/tr3d/ms/fog/super/fog_ms_tr3d-ff_sunrgbd-10class.py 2

# fog MS semi
# CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=10 PORT=5521 ./tools/dist_train.sh configs/tr3d/ms/fog/semi/fog_ms_tr3d-ff_sunrgbd-10class.py 2

# Normal MS - only visible
# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 PORT=5520 python tools/train.py configs/tr3d/ms/normal/ms_tr3d-ff_sunrgbd-10class.py

# Normal MS - only LWIR
# CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 PORT=5521 python tools/train.py configs/tr3d/ms/normal/ms_tr3d-ff_sunrgbd-10class.py


# imvote
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python tools/train.py configs/imvotenet/fog_imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py

# imvote fog

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 PORT=5524 python tools/test.py configs/imvotenet/fog_noise_imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py work_dirs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth --eval mAP

# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python tools/test.py configs/imvotenet/fog_imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py /home/dgkim/workspace/tr3d/work_dirs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth

# imvote lwir
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python tools/train.py configs/imvotenet/fog_imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 PORT=5524 python tools/test.py \
configs/imvotenet/fog_imvotenet_stage2_16x8_sunrgbd-3d-10class.py \
work_dirs/imvotenets/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210819_192851-1bcd1b97.pth --eval mAP
