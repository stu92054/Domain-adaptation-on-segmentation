export CUDA_VISIBLE_DEVICES="3"_
python source_trainer.py city --net drn_d_105 \
--train_img_shape 512 256 \
--batch_size 16 \
--epochs 40
#--resume ./train_output/city-train2Taipei-train_3ch/pth/MCD-normal-drn_d_105-17.pth.tar

