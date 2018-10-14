python ./src/infer.py \
 --img_path_file ./example_image/Taipei_test.txt \
 --eval_script ./src/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py \
 --city Taipei \
 --pretrained_weight ./pretrained/train_cscape_frontend.npy \
 --method GACA \
 --gt_dir ./example_image/labels/val/ \
 --weights_dir ./trained_weights/ \
 --output_dir ./train_results/ \
 --_format 'model' \
 --gpu 7 \
 --iter_lower 400 \
 --iter_upper 800

 
 
