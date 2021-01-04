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
parser.add_argument('--rate',type=float,default=1)
parser.add_argument('--len_k',type=int,default=5)
parser.add_argument('--final_k',type=int,default=5)
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
parser.add_argument('--imgtype',type=str,default='jpg')

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
  for folderi in range(args.final_k):
    if os.path.exists(args.output_dir+'/'+str(folderi)):
      os.system('rm -rf '+args.output_dir+'/'+str(folderi)+'/')
    os.system('mkdir '+args.output_dir+'/'+str(folderi))

@cuda.jit
def bitand(A,B):
  i,j=cuda.grid(2)
  A[i,j]=(A[i,j]+B[i,j])/2


if args.load=='':
  path=args.input_dir
  size=(100,100)
  filelist=glob(path+'/*.'+args.imgtype)
#print(filelist)
  model=joblib.load(args.model_name)
#litpic_num=0
  pic_num=0
  FL=len(filelist)
  FK=args.k
  if args.concat_k!='':
    ck=args.concat_k
    ks=ck.split(',')
    if len(ks)==1:
      FK+=int(ck)
    else:
      for ki in ks:
        FK+=int(ki)
  Features=np.zeros([FL,FK])
  current_k=args.k
#===================================
  for file in filelist:
    img=EX.rec(file,45,630,85,670,100)
    vec=np.zeros(FK)
    label=measure.label(img)
    ps=measure.regionprops(label)
    outpic=np.ones([100,100])
    for item in ps:
      if 1:
    #if item.area<avg_area*1.1 and item.area>avg_area*0.9:
    #  litpic_num+=1
        a,b,c,d=item.bbox
        P=draw(a,b,c,d,img,item.label,label)
        SP=cv2.resize(P,size)
        SP=np.reshape(SP,(1,10000))
        score=model.predict(SP)
        vec[int(score)]=1
        if args.concat_model!='': 
          models=args.concat_model.split(',')
          if len(models)==1:
            concat_model=joblib.load(args.concat_model)
            concat_score=concat_model.predict(SP)+current_k
          else:
            for mi in range(len(models)):
              concat_model=joblib.load(models[mi])
              concat_score=concat_model.predict(SP)+current_k
              current_k+=int(ks[mi])
          vec[int(concat_score)]=1
          current_k=args.k
    Features[pic_num]=vec
    pic_num+=1
    print(pic_num)   
  #print(vec)  
else:
  infile=open(args.load)
  content=infile.read().split('\n')
  content.remove('')
  FL=len(content)
  filelist=[]
  Features=np.zeros([FL,500])
  LenFeatures=np.zeros([FL,10])
  for i in range(FL):

    namesp=content[i].split('|')
    filelist.append(namesp[0])
    Fs=namesp[1].split(' ')
    Fs.remove('')
    #print(len(Fs))
    img=EX.rect(filelist[i])
    label=measure.label(img)
    ps=measure.regionprops(label)
    little_pic=0
    allarea=0
    for item in ps:
      allarea+=item.area
      little_pic+=1
    avg_area=allarea/little_pic
    Y=EX.bin(avg_area)
    
    for k in range(500):
      Features[i,k]=np.float32(Fs[k])
   
    for k in range(0,10):
      LenFeatures[i,k]=Y[k]
    
  
#================================================
if args.mode=='save':
  out_file=open(args.output,'w')
  for i in range(FL):
    out_file.write(filelist[i])
    out_file.write('|')
    for item in Features[i]:
      out_file.write(str(int(item)))
      out_file.write(' ')
    out_file.write('\n')

if args.mode=='little':
  #labels=np.zeros(FL)
 # create_dir()
  '''
  pca=PCA(0.99)
  Features=pca.fit_transform(Features)
  print(Features.shape)
  pcalog=open('pca.log','w')
  pcalog.write(str(Features.shape))
  pcalog.close()
  '''

  for i in range(FL):
    print(filelist[i],Features[i])
   # labels[i]=Features[i]

if args.mode=='train':
  kmeans = KMeans(n_clusters=args.pic_k,init='random',n_init=args.n_init,max_iter=args.iter)
  kmeans.fit(Features)
  joblib.dump(kmeans,args.output_model)

if args.mode=='test':
 # create_dir()
  model=SpectralClustering(n_clusters=args.pic_k,gamma=args.gamma)
  labels =model.fit_predict(Features)
  score = calinski_harabaz_score(Features,labels)
  
  print(labels)
  print(score)

  Lmodel=SpectralClustering(n_clusters=args.len_k,gamma=args.gamma)
  Llabels =Lmodel.fit_predict(LenFeatures)
  Lscore = calinski_harabaz_score(LenFeatures,Llabels)
  
  print(Llabels)
  print(Lscore)
  CF=np.zeros([FL,2])
  for i in range(FL):
    CF[i,0]=labels[i]
    CF[i,1]=Llabels[i]*args.rate

  #print(CF)
  
  Fmodel=SpectralClustering(n_clusters=args.final_k,gamma=args.gamma)
  Flabels =Fmodel.fit_predict(CF)
  Fscore = calinski_harabaz_score(CF,Flabels)
  
  print(Flabels)
  print(Fscore)

  create_dir()
#  L=len(filelist)
  for i in range(FL):
    cmd='cp '+filelist[i]+' '+args.output_dir+'/'+str(Flabels[i])+'/'
   # print(cmd)
    os.system(cmd)

if args.mode=='all':
  model=SpectralClustering(n_clusters=args.pic_k,gamma=args.gamma)
  #Features=preprocessing.scale(Features)
  '''
  pca=PCA(0.99)
  Features=pca.fit_transform(Features)
  print(Features.shape)
  pcalog=open('pca.log','w')
  pcalog.write(str(Features.shape))
  pcalog.close()
  '''
  #print(Features)
  labels =model.fit_predict(Features)
  score = calinski_harabaz_score(Features,labels)
  joblib.dump(model,args.output_model)
  print(labels)
  print(score)

  #labels=kmeans.fit_predict(Features)
#  '''
  create_dir()
  L=len(filelist)
  for i in range(L):
    cmd='cp '+filelist[i]+' '+args.output_dir+'/'+str(labels[i])+'/'
   # print(cmd)
    os.system(cmd)
#  '''
