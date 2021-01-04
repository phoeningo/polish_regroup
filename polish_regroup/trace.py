import cv2
from numba import cuda,jit
from glob import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ex import op as EX
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import argparse
from sklearn.metrics import calinski_harabaz_score


parser=argparse.ArgumentParser(description='T')
parser.add_argument('--k',type=int,default=10)
parser.add_argument('--gamma',type=float,default=0.1)
parser.add_argument('--input_dir',type=str,default='temp')
parser.add_argument('--output_dir',type=str,default='classes')
parser.add_argument('--model_name',type=str,default='S250.m')
parser.add_argument('--mode',type=str,default='train')
parser.add_argument('--n_init',type=int,default=10)
parser.add_argument('--iter',type=int,default=300)
parser.add_argument('--output',type=str,default='cut.f')
parser.add_argument('--load',type=str,default='')
parser.add_argument('--invert',type=str,default='T')
parser.add_argument('--box_size',type=int,default=64)
args=parser.parse_args()

def create():
  EX.create_newdir(args.output_dir)
  for folderi in range(args.k):
    if os.path.exists(args.output_dir+'/'+str(folderi)):
      os.system('rm -rf '+args.output_dir+'/'+str(folderi)+'/')
    os.system('mkdir '+args.output_dir+'/'+str(folderi))

def cpfile(train_filelist):
  L=len(train_filelist)
  for i in range(L):
    cmd='cp '+train_filelist[i]+' '+args.output_dir+'/'+str(labels[i])+'/'
   # print(cmd)
    os.system(cmd)


if args.load=='':
  i=0
  final_pic=0
  train_filelist=glob(args.input_dir+'/*.png')

  for files in train_filelist:
    proj=np.uint8(EX.lit(files))
  #  print(proj)
#  print(proj.shape)
    if args.invert=='T':
      proj=255-proj
  #  print(np.max(proj))
    x,y=proj.shape
   # cv2.imwrite('temp.png',proj)
    proj=proj.flatten().reshape(1,x*y)
    if i==0:
      final_pic=proj
    else:
      final_pic=np.concatenate((final_pic,proj))
    i=i+1
    print(i)
  #print(final_pic.shape)
  
  X=final_pic
else:
  infile=open(args.load)
  content=infile.read().split('\n')
  content.remove('')
  FL=len(content)
  filelist=[]
  Features=np.zeros([FL,10000])
  for i in range(FL):
    Fs=content[i].split(' ')
    Fs.remove('')
    for k in range(10000):
      Features[i,k]=np.float32(Fs[k])

#==============================



if args.mode=='save':
  FL=len(final_pic)
  out_file=open(args.output,'w')
  for i in range(FL):
    for item in final_pic[i]:
      out_file.write(str(int(item)))
      out_file.write(' ')
    out_file.write('\n')




if args.mode=='kmeans':
  model=KMeans(n_clusters=args.k,n_init=args.n_init,max_iter=args.iter)
  labels=model.fit_predict(final_pic)
  joblib.dump(model,args.model_name)
  print(labels)  

  L=len(train_filelist)
  create()
  cpfile(train_filelist)


if args.mode=='test':
  model=joblib.load(args.model_name)
 # '''
  EX.create_newdir(args.output_dir)
  for folderi in range(args.k):
    if os.path.exists(args.output_dir+'/'+str(folderi)):
      os.system('rm -rf '+args.output_dir+'/'+str(folderi)+'/')
    os.system('mkdir '+args.output_dir+'/'+str(folderi))
 # '''
  labels=model.predict(final_pic)
  L=len(train_filelist)
  for i in range(L):
    cmd='cp '+train_filelist[i]+' '+args.output_dir+'/'+str(labels[i])+'/'
   # print(cmd)
    os.system(cmd)

if args.mode=='all':
  model=SpectralClustering(n_clusters=args.k,gamma=args.gamma)
  print('begining')
 # create()
  labels=model.fit_predict(final_pic)
  joblib.dump(model,args.model_name)
  score = calinski_harabaz_score(final_pic,labels)
  print(labels,score)
#  cpfile(train_filelsit)


