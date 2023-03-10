# SkyCloudNet - ICIVC 2023 (#19)


###
This is the code for the submission __#19__  SkyCloud: A Neural Network-based Approach for Sky and Cloud Segmentation from Natural Images at __ICIVC 2023__. 

![cloudseg_example_2](https://user-images.githubusercontent.com/24622304/187810011-82a2c390-9074-4d8f-92e3-6b350c29d566.png)

The repository currently only contains the code for the evaluation. The training code is currently optimized for multi-GPU training and will be published at the time of publication. The code was tested on Ubuntu 22.04 LTS.

## Run the code
### Requirements
- Python: >3.2

- CUDA: 10.2

To install the required python packages run: 
```
pip install -r requirements.txt
```

### Data set and weights
The __SkyCloud__ data set and the pretrained weights are available at: https://osf.io/y69ah/?view_only=889215916ccb4c52a5971fffc6af0dda.

### Quick start 
Download the __SkyCloud__ data set. 
Open the config file ```config/config.yaml``` and add the path to the data set root folder at ```root_dataset```

Download the pretrained weights and save them to the ```weights``` folder.
Alternatively change the ```DIR``` folder to the location of the weights. 

Test the network:
```
python3 eval.py --cfg config/config.yaml
```
