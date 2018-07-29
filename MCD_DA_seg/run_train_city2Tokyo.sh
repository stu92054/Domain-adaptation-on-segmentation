export CUDA_VISIBLE_DEVICES="3"_
python adapt_trainer.py city Tokyo --net drn_d_105 \
--train_img_shape 512 256 \
--batch_size 8 \

