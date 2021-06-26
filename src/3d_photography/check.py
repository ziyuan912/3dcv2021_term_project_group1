import cv2
import os
import shutil
dir_path = 'image/'
new_dir_path = 'images/'
filename = os.listdir(dir_path)
filename = [i for i in filename if i.find('jpg') != -1]
count = 0
for i in filename:
    name = i
    i = i[6:9]
    if os.path.isfile('demo/img{}.jpg'.format(i)):
        shutil.move(dir_path + name, new_dir_path + name)
