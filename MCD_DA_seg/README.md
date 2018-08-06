# Maximum Classifier Discrepancy for Domain Adaptation with Semantic Segmentation Implemented by PyTorch

We use the code provided by [https://github.com/mil-tokyo/MCD_DA](https://github.com/mil-tokyo/MCD_DA) and additionally create data-loaders for our dataset.
  
***
## Installation
Use **Python 2.x**

First, you need to install PyTorch following [the official site instruction](http://pytorch.org/).

Next, please install the required libraries as follows;
```
pip install -r requirements.txt
```

## Usage

### Dataset

* Download [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
* Download [Synthia Dataset](http://synthia-dataset.com/download-2/)
	* download the subset "SYNTHIA-RAND-CITYSCAPES" 
* Download [Our Dataset](https://yihsinchen.github.io/segmentation_adaptation/#Dataset)
	* contains four subsets --- Taipei, Tokyo, Roma, Rio --- used as target domain (only testing data has annotations) 
* Check data-root-path in datasets.py
 
### Testing
Download the trained model (synthia-to-cityscapes w. dilated-ResNet @60-epoch):

```
$ cd MCD_DA_seg  
$ sh download_demo.sh
```

##### Infer the model: 

```
python adapt_tester.py city ./train_output/synthia-train2city-train_3ch/pth/MCD-normal-drn_d_105-res50-60.pth.tar
```

(you can use the script "run_test.sh")

Results will be saved under "./test_output/synthia-train2city-train_3ch---city-val/MCD-normal-drn_d_105-res50-60.tar"

##### Evaluation:
We replace the original evaluation code with the script provided by Cityscapes-Dataset. (use "run_eval.sh")

```
python ./cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py  
--gt {GroundTruth File for evalation} 
--pd ./test_output/synthia-train2city-train_3ch---city-val/MCD-normal-drn_d_105-60.tar/label/
```

### Training Examples 
- Dataset
    - Source: Synthia (synthia), Target: Cityscapes (city)
- Network
    - Dilated Residual Network (drn_d_105)

We train the model following the assumptions above;
```
python adapt_trainer.py synthia city --net drn_d_105
```

Trained models will be saved as "./train_output/synthia-train2city-train_3ch/pth/MCD-normal-drn_d_105-res50-EPOCH.pth.tar"

The training scripts for adapt from (A) Synthia dataset to Cityscapes dataset and (B) Cityscapes dataset to Our dataset are prepared in "run_train_syn2city.sh" and "run_train_city2ours.sh". 

For using our dataset "CrossCountries" as the Target domain, in above scripts, replace Dataset-Name with the subset name {Taipei, Tokyo, Roma, Rio}
 
## Reference codes
- [https://github.com/mil-tokyo/MCD_DA](https://github.com/mil-tokyo/MCD_DA)
