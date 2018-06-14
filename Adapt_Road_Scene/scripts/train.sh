python -u ./tools/train_adv.py \
--weight_path ./pretrained/my_frontend_16200.npy \
--city Taipei \
--src_data_path ./data/Cityscapes.txt \
--tgt_data_path ./data/Taipei.txt \
--method GACA \
--batch_size 2 \
--iter_size 4 \
--max_step 12000 \
--save_step 1000 \
--train_dir ./trained_weights/ \
--gpu 3 \
2>&1 | tee ./logfiles/Taipei_FullMethod.log
#--method FullMethod \
#--restore_path './trained_weights/GA/Taipei/model-2000' \
