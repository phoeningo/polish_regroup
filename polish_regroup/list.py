from glob import glob
import os
import numpy as np
import argparse

parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input_dir',type=str)
args=parser.parse_args()

filelist=glob(args.input_dir+'/*')
Flen=len(filelist)
Folders=np.zeros([Flen,1])
for i in range(Flen):
  files=glob(filelist[i]+'/*.png')
  Folders[i]=len(files)
#print(np.argsort(Folders,axis=0)[0])
#print(filelist[int(np.argsort(Folders,axis=0)[1])])
#print(filelist[29])
print('The largest set : %i'%int(np.max(Folders)))
print('The smallest set : %i'%int(np.min(Folders)))
