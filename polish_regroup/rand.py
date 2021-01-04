from glob import glob
import numpy as np
import sys
import os
import argparse
from ex import op as EX
parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input_dir',type=str)
parser.add_argument('--output_dir',type=str)
parser.add_argument('--rate',type=float)
args=parser.parse_args()
files=glob(args.input_dir+'/*')
EX.create_newdir(args.output_dir)


for it in files:
  ra=np.random.random(1)
  if ra<args.rate:
    cmd='cp '+it+' '+args.output_dir+'/'
    #print(cmd)
    os.system(cmd)


