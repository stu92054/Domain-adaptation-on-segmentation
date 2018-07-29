python -u ./tools/train_adv.py \
--weight_path ./pretrained/pretrained_vgg.npy \
--restore_path './pretrained/train_synthia_frontend' \
--city syn2real \
--src_data_path ./data/synthia_train.txt \
--tgt_data_path ./data/Cityscapes.txt \
--method GACA \
--batch_size 8 \
--iter_size 4 \
--start_step 0 \
--max_step 10000 \
--save_step 400 \
--train_dir ./trained_weights/ \
--gpu 5,6 \
2>&1 | tee ./logfiles/syn2real_GACA.log
