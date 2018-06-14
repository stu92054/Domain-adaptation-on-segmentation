python ./tools/infer.py \
 --eval_script ./tools/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py \
 --city Taipei \
 --load_npy 1 \
 --pretrained_weight ./pretrained/my_frontend_16200.npy \
 --method pretrained \
 --img_dir /media/VSlab2/BCTsai/Lab/datasets/NTHU_512/imgs/val/ \
 --gt_dir /media/VSlab2/BCTsai/Lab/datasets/NTHU_512/labels/val/ \
 --weights_dir ./trained_weights/ \
 --output_dir ./train_results/ \
 --_format 'model' \
 --gpu 0 \
 --iter_lower 2 \
 --iter_upper 2
 #--_format 'Tensor("lr:0", shape=(), dtype=float32)' \

 
 
