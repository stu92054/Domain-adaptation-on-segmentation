mkdir pretrained
sh scripts/cmd_for_DL.sh 1HkAewAjxyQXF8jrI-7lfNdMTA5AV6NVL pretrained/pretrained_vgg.npy 
sh scripts/cmd_for_DL.sh 1euYkvtI0Op99mjgv2Nl4eDKk6t2_zLu2 pretrained/train_cscape_frontend.npy 
sh scripts/cmd_for_DL.sh 1noylQcOyXGB_QPCC0T994QQP-8I4li4z pretrained/train_synthia_frontend.zip 
uzip pretrained/train_synthia_frontend.zip -d pretrained