import os
import cv2
import mrcfile
import sys
import argparse
import numpy as np
from numba import cuda,jit
import time
import math
from skimage import measure,exposure,morphology
from ex import op as EX
from glob import glob

parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input_dir',type=str)
parser.add_argument('--output_dir',type=str)
args=parser.parse_args()


def draw(miny,minx,maxy,maxx,pic,tag,label):
  subpic=np.zeros([maxy-miny,maxx-minx])
  for i in range(miny,maxy):
    for j in range(minx,maxx):
      if label[i,j]==tag:
        subpic[i-miny,j-minx]=pic[i,j]
  return subpic.T

@cuda.jit
def bitand(A,B):
  i,j=cuda.grid(2)
  A[i,j]=(A[i,j]+B[i,j])/2

path=args.input_dir
size=(100,100)
filelist=glob(path+'/*.png')
EX.create_newdir(args.output_dir)
litpic_num=0
filenum=0
#===================================
for file in filelist:
  filenum+=1
  print(str(filenum)+' / '+str(len(filelist))+' files.')
  img=EX.rec(file,45,630,85,670,100)
  label=measure.label(img)
  ps=measure.regionprops(label)
  '''
  num=0
  sum_area=0
  for item in ps:
    num=num+1
    sum_area+=item.area
  avg_area=sum_area/num 
  '''
  outpic=np.ones([100,100])
  for item in ps:
    if 1: 
   #if item.area<avg_area*1.1 and item.area>avg_area*0.9:
      litpic_num+=1
      a,b,c,d=item.bbox
      P=draw(a,b,c,d,img,item.label,label)
      print(P.shape)
      #SP=cv2.resize(P,size)
      cv2.imwrite(args.output_dir+'/'+str(litpic_num)+'.png',P)
     # print(litpic_num)
  