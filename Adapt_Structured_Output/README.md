# Learning to Adapt Structured Output Space for Semantic Segmentation
We use the code provided by [https://github.com/wasidennis/AdaptSegNet](https://github.com/wasidennis/AdaptSegNet) and additionally create data-loaders for our dataset.

----------


## Installation
* Use Python2

## Dataset
Download the fowllowing dataset and put them in the `data` folder

* Download [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
* Download [Synthia Dataset]()
* Download [Our Dataset]()
	* contains four subsets --- Taipei, Tokyo, Roma, Rio --- as target domain (only testing data has annotations) 

## Testing
* Download and testing [the trained model](https://drive.google.com/open?id=1MKnzjzl0aovlUH1NDK_6qw8LRB1AoZFa) and put it in the model folder

* Test the model and results will be saved in the `result` folder
```python evaluate_cityscapes.py --restore-from ./model/Synthia2cityscapes.pth```

* Compute the IoU on Cityscapes (thanks to the code from [VisDA Challenge](http://ai.bu.edu/visda-2017/))
```python compute_iou.py ./data/Cityscapes/data/gtFine/val result/cityscapes```

	The demo model is sythia-to-cityscapes, and results will be saved in the `./result/` folder. Also, it shows evaluated performance. (the evaluation code is provided by Cityscapes-dataset) 

## Training Examples

* Train the synthia-to-Cityscapes model (multi-level)
```python train_synthia2cityscapes_multi.py --snapshot-dir ./snapshots/GTA2Cityscapes_multi --lambda-seg 0.1 --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001```

##Reference code
[https://github.com/wasidennis/AdaptSegNet](https://github.com/wasidennis/AdaptSegNet)
