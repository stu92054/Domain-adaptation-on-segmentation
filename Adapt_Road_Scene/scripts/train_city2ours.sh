python -u ./tools/train_adv.py \
--weight_path ./pretrained/train_cscape_frontend.npy \
--city Taipei \
--src_data_path ./data/Cityscapes.txt \
--tgt_data_path ./data/Taipei.txt \
--method FullMethod \
--batch_size 8 \
--iter_size 4 \
--start_step 0 \
--max_step 3000 \
--save_step 400 \
--train_dir ./trained_weights/ \
--gpu 5,6 \
2>&1 | tee ./logfiles/Taipei_FullMethod.log
