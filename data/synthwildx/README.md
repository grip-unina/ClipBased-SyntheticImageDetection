# SynthWildX

SynthWildX is a dataset of images collected from the social network X.
It includes 500 real images and 1,500 synthetic images from three different generators DALL·E 3, Midjourney and Firefly.
To understand which generator was used to create a specific image, we relied on tags and annotations present on the relative post.

To download the images directly from X, execute the python script `download_synthwildx.py`.

If you plan to use the dataset, please cite the paper [Raising the Bar of AI-generated Image Detection with CLIP](https://arxiv.org/abs/2312.00195).


```
@inproceedings{cozzolino2023raising,
  author={Davide Cozzolino and Giovanni Poggi and Riccardo Corvi and Matthias Nießner and Luisa Verdoliva},
  title={Raising the Bar of AI-generated Image Detection with CLIP}, 
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024}
}
```