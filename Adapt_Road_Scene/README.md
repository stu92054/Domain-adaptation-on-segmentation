# No More Discrimination: Cross City Adaptation of Road Scene Segmenters Implemented by Tenosrflow
Paper link: [https://arxiv.org/abs/1704.08509](https://arxiv.org/abs/1704.08509) 

***

## Intro 
Tensorflow implementation of the paper for adapting semantic segmentation from the (A) Synthia dataset to Cityscapes dataset and (B) Cityscapes dataset to Our dataset.

## Installation
* Use Tensorflow version-1.1.0 with Python2

## Dataset

* Download [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
* Download [Synthia Dataset](http://synthia-dataset.com/download-2/)
	* download the subset "SYNTHIA-RAND-CITYSCAPES" 
* Download [Our Dataset](https://yihsinchen.github.io/segmentation_adaptation/#Dataset)
	* contains four subsets --- Taipei, Tokyo, Roma, Rio --- used as target domain (only testing data has annotations) 
* Change the data path in files under folder "./data"

## Testing

* Download and testing the trained model 

	```	
	cd fcns-wild
	sh scripts/download_demo.sh
	sh scripts/infer_city2Taipei.sh 	
	```

	The demo model is cityscapes-to-Taipei(a subset of our dataset), and results will be saved in the `./train_results/` folder. Also, it shows evaluated performance. (the evaluation code is provided by Cityscapes-dataset) 


## Training Examples
* Download the pretrained weights (model trained on source)
	```
	sh download_src.sh	
	```
* Train the Cityscapes-to-Ours{city} model 

	```
	python ./src/train_adv.py \
		--weight_path ./pretrained/train_cscape.npy \
		--city {city_name} \
		--src_data_path ./data/Cityscapes.txt \
		--tgt_data_path ./data/{city_name}.txt \
		--method FullMethod \
	```

	The training scripts for adapt from (A) Synthia dataset to Cityscapes dataset and (B) Cityscapes dataset to Our dataset are prepared in "scripts/run_train_syn2city.sh" and "scripts/run_train_city2ours.sh". 

 
## Reference codes
[https://github.com/MarvinTeichmann/tensorflow-fcn](https://github.com/MarvinTeichmann/tensorflow-fcn)
