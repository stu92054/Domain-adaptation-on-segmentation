python ./tools/infer.py \
 --img_path_file ./data/Taipei_val.txt \
 --eval_script ./tools/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py \
 --city Taipei \
  --method GA \
 --pretrained_weight ./pretrained/train_cscape_frontend.npy \
 --gt_dir XXX/NTHU_512/labels \
 --output_dir ./train_results/ \
 --weights_dir ./trained_weights/ \
 --_format 'model' \
 --gpu 2 \
 --iter_lower 1600 \
 --iter_upper 2000
 
 
 
