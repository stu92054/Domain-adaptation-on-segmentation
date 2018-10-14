python -u ./src/train_adv.py \
--weight_path ./pretrained/train_cscape_frontend.npy \
--city Roma \
--src_data_path ./data/Cityscapes.txt \
--tgt_data_path ./data/Roma.txt \
--method GACA \
--batch_size 4 \
--iter_size 4 \
--max_step 2000 \
--save_step 200 \
--train_dir ./trained_weights/ \
--gpu 5,6 \
2>&1 | tee ./logfiles/Roma_GACA.log
