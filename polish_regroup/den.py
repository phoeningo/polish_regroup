import os
import cv2
import mrcfile
import sys
import argparse
import numpy as np
from numba import cuda,jit
from sklearn.cluster import KMeans
import time
import math
from skimage import measure,exposure,morphology
from ex import op as EX
from glob import glob
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.cluster import SpectralClustering 
from sklearn.metrics import calinski_harabaz_score


parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input_dir',type=str,default='RAND')
parser.add_argument('--k',type=int,default=500)
parser.add_argument('--concat_k',type=str,default='')
parser.add_argument('--pic_k',type=int,default=10)
parser.add_argument('--model_name',type=str,default='V250.m')
parser.add_argument('--concat_model',type=str,default='')
parser.add_argument('--train_model',type=str,default='train.m')
parser.add_argument('--output_model',type=str,default='train.m')
parser.add_argument('--output_dir',type=str,default='')
parser.add_argument('--mode',type=str,default='test')
parser.add_argument('--n_init',type=int,default=10)
parser.add_argument('--iter',type=int,default=300)
parser.add_argument('--eps',type=float,default=1)
parser.add_argument('--min_samples',type=int,default=10)
parser.add_argument('--gamma',type=float,default=0.5)
parser.add_argument('--metric',type=str,default='manhattan')
parser.add_argument('--load',type=str,default='')
parser.add_argument('--output',type=str,default='Features.log')
args=parser.parse_args()


def draw(miny,minx,maxy,maxx,pic,tag,label):
  subpic=np.zeros([maxy-miny,maxx-minx])
  for i in range(miny,maxy):
    for j in range(minx,maxx):
      if label[i,j]==tag:
        subpic[i-miny,j-minx]=pic[i,j]
  return subpic.T



def create_dir():
  EX.create_newdir(args.output_dir)
  for folderi in range(args.pic_k):
    if os.path.exists(args.output_dir+'/'+str(folderi)):
      os.system('rm -rf '+args.output_dir+'/'+str(folderi)+'/')
    os.system('mkdir '+args.output_dir+'/'+str(folderi))

@cuda.jit
def bitand(A,B):
  i,j=cuda.grid(2)
  A[i,j]=(A[i,j]+B[i,j])/2


if args.load=='':
  model=KMeans(n_clusters=args.pic_k)
  path=args.input_dir
  size=(100,100)
  filelist=glob(path+'/*.png')
  L=len(filelist)
  LS=np.zeros([L,1])
  filei=0
  for file in filelist:
    img=EX.rect(file)

    label=measure.label(img)
    ps=measure.regionprops(label)

    little_pic=0
    allarea=0
    for item in ps:
      allarea+=item.area
      little_pic+=1
    avg_area=allarea/little_pic
    LS[filei]=avg_area
    filei+=1
  
  labels=model.fit_predict(LS)
  create_dir()  

  for i in range(L):
    cmd='cp '+filelist[i]+' '+args.output_dir+'/'+str(labels[i])+'/'
   # print(cmd)
    os.system(cmd)


