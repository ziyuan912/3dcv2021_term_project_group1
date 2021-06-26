import os
from google.colab import files
from IPython.display import HTML
from skimage import io, transform
from glob import glob
import numpy as np

epochs =  40
image_width =  400

## utility function
def image2datadir(image_path):
    img = io.imread(image_path)
    oh, ow, oc = img.shape
    ratio = image_width / ow
    down_img = transform.resize(img, (int(oh * ratio), (ow * ratio)),anti_aliasing=True)
    down_img *= 255
    down_img = down_img.astype(np.uint8)
    _, filename = os.path.split(image_path)
    io.imsave(os.path.join('data/demo/images',filename),down_img)
    os.remove(image_path)

# clear previous run directory and prepre new one
# !rm -rf data/demo
# !rm -rf data/runs
# !rm -rf data/upload
# !mkdir -p data/demo
# !mkdir -p runs 
# prepare the dataset

image_files = sorted(glob('./data/test/images/*.jpg')) 
print(images)
if len(image_files) < 12:
    print("Failed, You must contain at least 12 images.")
else:
    os.mkdir('data/demo/sparse')
for f in image_files:
    image2datadir(f)
get_ipython().system_raw('apt install colmap')
print('Run SFM')
!colmap feature_extractor --database_path data/demo/database.db --image_path data/demo/images --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE --SiftExtraction.use_gpu=false 
!colmap exhaustive_matcher --database_path data/demo/database.db  --SiftMatching.use_gpu=false
!colmap mapper --database_path data/demo/database.db --image_path data/demo/images --Mapper.ba_refine_principal_point 1 --Mapper.num_threads 2 --Mapper.extract_colors 0 --export_path data/demo/sparse
!colmap image_undistorter --image_path data/demo/images --input_path data/demo/sparse/0 --output_path data/demo/dense --output_type COLMAP
# we have to import colmap_runner after repo succesfull load
from utils.colmap_runner import load_colmap_data, save_poses
poses, pts3d, perm, hwf_cxcy = load_colmap_data('data/demo')
save_poses('data/demo', poses, pts3d, perm, hwf_cxcy)