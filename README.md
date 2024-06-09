# ClipBased-SyntheticImageDetection

[![Github](https://img.shields.io/badge/Github%20page-222222.svg?style=for-the-badge&logo=github)](https://grip-unina.github.io/ClipBased-SyntheticImageDetection/)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2312.00195v2)
[![GRIP](https://img.shields.io/badge/-GRIP-0888ef.svg?style=for-the-badge)](https://www.grip.unina.it)

This is the official repository of the paper:
[Raising the Bar of AI-generated Image Detection with CLIP](https://arxiv.org/abs/2312.00195v2).

Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner, and Luisa Verdoliva.

## Overview

The aim of this work is to explore the potential of pre-trained vision-language models (VLMs) for universal detection of AI-generated images. We develop a lightweight detection strategy based on CLIP features and study its performance in a wide variety of challenging scenarios. We find that, contrary to previous beliefs, it is neither necessary nor convenient to use a large domain-specific dataset for training. On the contrary, by using only a handful of example images from a single generative model, a CLIP-based detector exhibits surprising generalization ability and high robustness across different architectures, including recent commercial tools such as Dalle-3, Midjourney v5, and Firefly. We match the state-of-the-art (SoTA) on in-distribution data and significantly improve upon it in terms of generalization to out-of-distribution data (+6% AUC) and robustness to impaired/laundered data (+13%).

## License

Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.

## Test Code
Before using the code, download the weights:

```
git lfs pull
```

The `main.py` script requires as input a CSV file with the list of images to analyze.
The input CSV file must have a 'filename' column with the path to the images.
The code outputs a CSV file with the LLR score for each image.
If LLR>0, the image is detected as synthetic.

The `compute_metrics.py` script can be used to evaluate metrics.
In this case, the input CSV file must also include the 'typ' column with a value equal to 'real' for real images.


### Script 
In order to use the script the follwing packages should be installed:

	* tqdm
	* scikit-learn
	* pillow
	* yaml
	* pandas
	* torchvision
	* torch
	* timm>=0.9.10
	* huggingface-hub>=0.23.0
	* open_clip_torch

The test can be executed as follows:

```
python main.py --in_csv /path/input/csv --out_csv /path/output/csv --device 'cuda:0'
```

To get the results on Commercial Tools generators:
1) Firstly, download the synthbuster dataset using the following command:
```
cd data; bash synthbuster_download.sh; cd ..
```

2) Then, run the `main.py` script as follows: 
```
python main.py --in_csv data/commercial_tools.csv --out_csv out.csv --device 'cuda:0'
```

3) Finally, calculate the AUC metrics:
```
python compute_metrics.py --in_csv data/commercial_tools.csv --out_csv out.csv --metrics auc --save_tab auc_table.csv
```

### Docker
To build the docker image, run the following command:
```
docker build -t clipdet . -f Dockerfile
```

To get the results on Commercial Tools generators, it can be launched as follows:
```
docker run --runtime=nvidia --gpus all -v ${PWD}/data:/data_in -v ${PWD}/:/data_out clipdet \
    --in_csv /data_in/commercial_tools.csv --out_csv /data_out/out_docker.csv --device 'cuda:0'
```


## Bibtex 

```
@inproceedings{cozzolino2023raising,
  author={Davide Cozzolino and Giovanni Poggi and Riccardo Corvi and Matthias Nießner and Luisa Verdoliva},
  title={{Raising the Bar of AI-generated Image Detection with CLIP}}, 
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024},
}
```