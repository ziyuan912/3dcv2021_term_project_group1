# 3DCV 2021 Term Project Group 1

R09922074 潘奕廷 R09922031 黃子源 R09944030 高晟瑋

## Introduction
This project applies view synthesis methods to an image or motion video to get 3D views and place the result in the ideal region through augmented reality, which makes a cool result like the magic effect in Harry Potter films. 

## Prerequisites
* Python 3.8
* OpenCV-Python
* Open3D
* Shapely
* Matplotlib
* Numpy

## Usage
1. Clone and unzip files
```
git clone https://github.com/ziyuan912/3dcv2021_term_project_group1.git

cd 3dcv2021_term_project_group1

unzip ./src/testcases/1/sfm_data.zip ./testcases/1/

unzip ./src/testcases/2/sfm_data.zip ./testcases/2/
```
2. Test Nex method result
```
cd src

python test_nex.py
```
The result will save under the **src/testcases/1/final_result** folder.

4. Test 3D photography result
```
cd src

python test_photography.py
```
The result will save under the **src/testcases/2/final_result** folder.

## Training
We take **Nex[1]** and **3D photography[2]** as our view synthesis models. For retraining our final results, please see the README files under the both **src/nex** folder and **src/3d_photography** folder.

## Implementation
For detail about our implementation, please check out the [report](https://github.com/ziyuan912/3dcv2021_term_project_group1/blob/main/Report.pdf) file.

## Reference
[[1] Suttisak Wizadwongsa, et al. NeX: Real-time View Synthesis with Neural Basis Expansion. CVPR 2021](https://nex-mpi.github.io/)
[[2] Shih, et al. 3d photography using context-aware layered depth inpainting. CVPR, 2020](https://shihmengli.github.io/3D-Photo-Inpainting/)
