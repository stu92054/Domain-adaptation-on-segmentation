python ./src/infer.py \
 --img_path_file ./data/cscape_val.txt \
 --eval_script ./src/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py \
 --city syn2real \
 --pretrained_weight ./pretrained/train_cscape_frontend.npy \
 --method GACA \
 --gt_dir XXX/gtFine/val/ \
 --weights_dir ./trained_weights/ \
 --output_dir ./train_results/ \
 --_format 'model' \
 --gpu 7 \
 --iter_lower 800 \
 --iter_upper 1200
 
 
