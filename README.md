# LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection

## LAA-Net 논문 기반 개선 아이디어 적용 model

## 기존 LAA-Net
![alt text](./demo/laa_net.png?raw=true)
This is an official implementation for LAA-Net! [[Paper](https://arxiv.org/pdf/2401.13856.pdf)]


|LAA-Net | FF++ |  CDF  |    DFW     |     DFD    |     DFDC   |
|--------|------|-------|------------|------------|------------|
|w/ BI|<table><thead><tr><th>AUC</th></tr></thead><tbody><tr><td>99.95</td></tr></tbody></table>|<table><thead><tr><th>AUC</th><th>AP</th></tr></thead><tbody><tr><td>86.28</td><td>91.93</td></tr></tbody></table>|<table><thead><tr><th>AUC</th><th>AP</th></tr></thead><tbody><tr><td>57.13</td><td>56.89</td></tr></tbody></table>|<table><thead><tr><th>AUC</th><th>AP</th></tr></thead><tbody><tr><td>99.51</td><td>99.80</td></tr></tbody></table>|<table><thead><tr><th>AUC</th><th>AP</th></tr></thead><tbody><tr><td>69.69</td><td>93.67</td></tr></tbody></table>|
|w/ SBI|<table><thead><tr><th>AUC</th></tr></thead><tbody><tr><td>99.96</td></tr></tbody></table>|<table><thead><tr><th>AUC</th><th>AP</th></tr></thead><tbody><tr><td>95.40</td><td>97.64</td></tr></tbody></table>|<table><thead><tr><th>AUC</th><th>AP</th></tr></thead><tbody><tr><td>80.03</td><td>81.08</td></tr></tbody></table>|<table><thead><tr><th>AUC</th><th>AP</th></tr></thead><tbody><tr><td>98.43</td><td>99.40</td></tr></tbody></table>|<table><thead><tr><th>AUC</th><th>AP</th></tr></thead><tbody><tr><td>86.94</td><td>97.70</td></tr></tbody></table>|

## Recommended Environment
*For experiment purposes, we encourage the installment of the following libraries. Both Conda or Python virtual env should work.*

* CUDA: 11.4
* [Python](https://www.python.org/): >= 3.8.x
* [PyTorch](https://pytorch.org/get-started/previous-versions/): 1.8.0
* [TensorboardX](https://github.com/lanpa/tensorboardX): 2.5.1
* [ImgAug](https://github.com/aleju/imgaug): 0.4.0
* [Scikit-image](https://scikit-image.org/): 0.17.2
* [Torchvision](https://pytorch.org/vision/stable/index.html): 0.9.0
* [Albumentations](https://albumentations.ai/): 1.1.0


## LAA-Net Pre-trained Models
* 📌 *The pre-trained weights of using BI and SBI can be found [here](https://www.dropbox.com/scl/fo/dzmldaytujdeuky69d5x1/AIJrH2mit1hxnl1qzavM3vk?rlkey=nzzliincrfwejw2yr0ovldru1&st=z8ds7il7&dl=0)!*

## Quickstart
1. **Preparation**

    1. ***Prepare environment***

        Installing main packages as the recommended environment.
    
    2. ***Prepare dataset***
        
        1. Downloading [FF++](https://github.com/ondyari/FaceForensics) *Original* dataset for training data preparation. Following the original split convention, it is firstly used to randomly extract frames and facial crops:
            ```
            python package_utils/images_crop.py -d {dataset} \
            -c {compression} \
            -n {num_frames} \
            -t {task}
            ```
            (*This script can also be utilized for cropping faces in other datasets such as [CDF](https://github.com/yuezunli/celeb-deepfakeforensics), [DFD](https://blog.research.google/2019/09/contributing-data-to-deepfake-detection.html), [DFDC](https://ai.meta.com/datasets/dfdc/) for cross-evaluation test. You do not need to run crop for [DFW](https://github.com/deepfakeinthewild/deepfake-in-the-wild) as the data is already preprocessed*).
            
            | Parameter | Value | Definition  |
            | --- | --- | --- |
            | -d | Subfolder in each dataset. For example: *['Face2Face','Deepfakes','FaceSwap','NeuralTextures', ...]*| You can use one of those datasets.|
            | -c | *['raw','c23','c40']*| You can use one of those compressions|
            | -n | *128*  | Number of frames (*default* 32 for val/test and 128 for train) |
            | -t | *['train', 'val', 'test']* | Default train|
            
            These faces cropped are saved for online pseudo-fake generation in the training process, following the data structure below:
            
            ```
            ROOT = '/data/deepfake_cluster/datasets_df'
            └── Celeb-DFv2
                └──...
            └── FF++
                └── c0
                    ├── test
                    │   └── frames
                    │       └── Deepfakes
                    |           ├── 000_003
                    |           ├── 044_945
                    |           ├── 138_142
                    |           ├── ...
                    │       ├── Face2Face
                    │       ├── FaceSwap
                    │       ├── NeuralTextures
                    │       └── original
                    |   └── videos
                    ├── train
                    │   └── frames
                    │       └── aligned
                    |           ├── 001
                    |           ├── 002
                    |           ├── ...  
                    │       └── original
                    |           ├── 001
                    |           ├── 002
                    |           ├── ...
                    |   └── videos
                    └── val
                        └── frames
                            ├── aligned
                            └── original
                        └── videos
            ```
        
        2. Downloading **Dlib** [[68]](https://github.com/davisking/dlib-models) [[81]](https://github.com/codeniko/shape_predictor_81_face_landmarks) facial landmarks detector pretrained and place into ```/pretrained/```. whereas the *68* and *81* will be used for the *BI* and *SBI* synthesis, respectively.

        3. Landmarks detection and alignment. At the same time, a folder for aligned images (```aligned```) is automatically created with the same directory tree as the original one. After completing the script running, a file that stores metadata information of the data is saved at ```processed_data/c0/{SPLIT}_<n_landmarks>_FF++_processed.json```.
            ```
            python package_utils/geo_landmarks_extraction.py \
            --config configs/data_preprocessing_c0.yaml \
            --extract_landmarks \
            --save_aligned
            ```
        4. (Optional) Finally, if using BI synthesis, for the online pseudo-fake generation scheme, 30 similar landmarks are searched for each facial query image beforehand.
            ```
            python package_utils/bi_online_generation.py \
            -t search_similar_lms \
            -f processed_data/c0/{SPLIT}_68_FF++_processed.json 
            ```
            
            *The final annotation file for training is created as* ```processed_data/c0/dynamic_{SPLIT}BI_FF.json```
        
2. **Training script**

    We offer a number of config files for specific data synthesis. 
    With *BI*, open ```configs/efn4_fpn_hm_adv.yaml```, please make sure you set ```TRAIN: True``` and ```FROM_FILE: True``` and run:
    ```
    ./scripts/train_efn_adv.sh
    ```

    Otherwise, with *SBI*, with the config file ```configs/efn4_fpn_sbi_adv.yaml```:
    ```
    ./scripts/efn_sbi.sh
    ```

3. **Testing script**

    For *BI*, open ```configs/efn4_fpn_hm_adv.yaml```, with ```subtask: eval``` in the *test* section, we support evaluation mode, please turn off ```TRAIN: False``` and ```FROM_FILE: False``` and run:
    ```
    ./scripts/test_efn.sh
    ```
    Otherwise, for *SBI*
    ```
    ./scripts/test_sbi.sh
    ```
    > ⚠️ *Please make sure you set the correct path to your download pre-trained weights in the config files.*

    > ℹ️ *Flip test can be used by setting ```flip_test: True```*
    
    > ℹ️ *The mode for single image inference is also provided, please set ```sub_task: test_image``` and pass an image path as an argument in test.py*


## model 개선 아이디어

1. **취약점 탐지 알고리즘 개선**

edge 검출에 이용되는 필터 중 하나인 Laplacian filter 연산을 이용하여 Heatmap에서 보다 정확한 취약 픽셀 탐지 가능성을 향상

<div align="center">
  <img src="./img/blend.png" alt="취약점 탐지 알고리즘">
</div>

***

2. **Heatmap 계산 알고리즘 개선**

단일 sigma가 아닌 다중 sigma를 이용한 계산 결과에 가중치를 곱한 값을 더하여 작은 sigma는 경계의 세밀한 부분을 강조하고 큰 sigma는 더 넓은 블렌딩 영역을 커버

<div align="center">
  <img src="./img/heatmap.png" alt="Heatmap 계산 알고리즘">
</div>


## Acknowledge
We acknowledge the excellent implementation from [mmengine](https://github.com/open-mmlab/mmengine), [BI](https://github.com/AlgoHunt/Face-Xray), and [SBI](https://github.com/mapooon/SelfBlendedImages).


## LICENSE
This software is © University of Luxembourg and is licensed under the snt academic license. See [LICENSE](NOTICE)


## CITATION
Please kindly consider citing our papers in your publications.

```
@InProceedings{Nguyen_2024_CVPR,
    author    = {Nguyen, Dat and Mejri, Nesryne and Singh, Inder Pal and Kuleshova, Polina and Astrid, Marcella and Kacem, Anis and Ghorbel, Enjie and Aouada, Djamila},
    title     = {LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17395-17405}
}
```
