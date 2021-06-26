from moviepy.editor import ImageSequenceClip
import cv2
import os
import numpy as np

filename = os.listdir("demo/")
filename = sorted(filename)
print(len(filename))
seq = []
for f in filename:
    img = cv2.imread('demo/' + f)
    myimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite('demo3/' + f, myimg)
    #print(f, img.shape)
    seq.append(img)
clip = ImageSequenceClip(seq, fps=40)
clip.write_videofile('video/output2.mp4', fps=40)