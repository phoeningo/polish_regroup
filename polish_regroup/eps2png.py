from PIL import Image
import numpy as np
from glob import glob
import cv2
import argparse
from numba import cuda,jit
from ex import op as EX
parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input',type=str)
parser.add_argument('--mode',type=str,default='full')
parser.add_argument('--eps_output',type=str)
parser.add_argument('--subeps_output',type=str)
parser.add_argument('--png_output',type=str)
parser.add_argument('--subpng_output',type=str)
parser.add_argument('--box_size',type=int,default=64)
parser.add_argument('--major',type=int,default=10)
args=parser.parse_args()

@cuda.jit
def bw(I,t):
  i,j,=cuda.grid(2)
  if I[i,j]<t:
    I[i,j]=0
  else:
    I[i,j]=255




filelist=glob(args.input+'/*.eps')
if args.mode=='subpng':
  EX.create_newdir(args.subpng_output)
  Folders=glob(args.input+'/*')
 # print(Folders)
  for folder in Folders:
    subfoldername=args.subpng_output+'/'+folder.split('/')[1]
    print(subfoldername)
    EX.create_newdir(subfoldername)
    filelist=glob(folder+'/*.eps')
    IMG=np.zeros([args.box_size,args.box_size],dtype=np.uint8)
    for file in filelist:
      name=file.split('/')
     # print(name)
     # name=name[1].split('.')[0]
      img=Image.open(file)
      savename=args.subpng_output+'/'+name[1]+'/'+name[2].split('.')[0]+'.png'
      #print(savename)
      img.save(savename)


if args.mode=='write':
  EX.sp_eps(file,args.eps_output)


if args.mode=='subeps':
  EX.create_newdir(args.subeps_output)
  for file in filelist:
    traces,fn=EX.get_traces(file)
    EX.extract_traces(traces,args.subeps_output,fn,args.box_size)


if args.mode=='avg':
  EX.create_newdir(args.subpng_output)
  Folders=glob(args.input+'/*')
  for folder in Folders:
 #   subfoldername=args.subpng_output+'/'+folder.split('/')[1]
 #   print(subfoldername)
 #   EX.create_newdir(subfoldername)
    filelist=glob(folder+'/*.eps')
    sublen=len(filelist)
    IMG=np.zeros([args.box_size,args.box_size])
    for file in filelist:
      name=file.split('/')
     # print(name)
     # name=name[1].split('.')[0]
      img=Image.open(file)
      imgdata=EX.single(np.asarray(img))
      inv=EX.inv(imgdata)
      EX.add(inv/sublen,IMG)
      IMIN=np.min(IMG)
      IMAX=np.max(IMG)
      IMG=255*(IMG-IMIN)/(IMAX-IMIN)
    imname=args.subpng_output+'/'+folder.split('/')[1]+'_avg.png'
    print(imname)
    cv2.imwrite(imname,IMG)
      
if args.mode=='t':
  filelist=glob(args.input+'/*.png')
  tempi=0
  for file in filelist:
    tempi+=1
    I=Image.open(file)
    
    img=np.array(np.asarray(I),dtype=np.uint8)
    blockdim=img.shape
    bw[blockdim,1](img,np.uint8(50))
    cv2.imwrite('test'+str(tempi)+'.png',img)

