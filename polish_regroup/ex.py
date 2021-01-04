from skimage import io,filters
from skimage import color
from numba import cuda,jit
import os

import numpy as np


@cuda.jit
def bw(I,t):
  i,j,=cuda.grid(2)
  if I[i,j]<t:
    I[i,j]=0
  else:
    I[i,j]=255


def create_newdir(path):
  if os.path.exists(path):
    os.system('rm -rf '+path+'/')
  os.system('mkdir '+path)


class op(object):
  @staticmethod
  def add(simg,img):
    sx,sy=simg.shape
    x,y=img.shape
    fx,fy=min(sx,x),min(sy,y)
   # print(fx,fy)
    for i in range(fx):
      for j in range(fy):
        img[i+max(0,int((x-fx)/2)),j+max(0,int((y-fy)/2))]+=simg[i+max(0,int((sx-fx)/2)),j+max(0,int((sy-fy)/2))]

  @staticmethod
  def binimg(img):
    x,y=img.shape
    blockdim=x,y
    griddim=1
    bw[blockdim,griddim](img,50)

  @staticmethod
  def inv(img):
    return np.max(img)-img

  @staticmethod
  def rect(path):
    img=io.imread(path)
    img=img[:,:,2]
    inv_img=255-img  
    x,y=img.shape
    blockdim=x,y
    griddim=1
    bw[blockdim,griddim](inv_img,50)
    rec=np.float32(inv_img[150:650,100:600])
    return rec

  @staticmethod
  def rec(path,a,b,c,d,t):
    img=io.imread(path)
    img=img[:,:,2]
    inv_img=255-img  
    x,y=img.shape
    blockdim=x,y
    griddim=1
    bw[blockdim,griddim](inv_img,t)
    rec=np.float32(inv_img[a:b,c:d])
    return rec


  @staticmethod
  def lit(path):
    img=io.imread(path)
    if len(img.shape)==3:
      img=img[:,:,2]
    return img

  @staticmethod
  def single(img):
    if len(img.shape)==3:
      img=img[:,:,2]
    return img

  @staticmethod
  def create_newdir(path):
    if os.path.exists(path):
      os.system('rm -rf '+path+'/')
    os.system('mkdir '+path)

  @staticmethod
  def create_dir(path):
    if os.path.exists(path):
      return
    os.system('mkdir '+path)


  @staticmethod
  def bin(x):
    X=int(x)
   # B=np.zeros(10)
    Y=np.zeros(10)
    L=0
    for i in range(10):
      Y[i]=X%2
      X=int(X/2)
    return Y

  @staticmethod
  def sp_eps(file,outdir):
    s=open(file)
    filedir=file.split('.eps')
    filename=filedir[0].split('/')
    name=outdir+'/'+filename[len(filename)-1]+'_s.eps'
    print(name)
    out=open(name,'w')
    con=s.read()

    temp1=con.split('fill\nstroke\nnewpath\n')
    len1=len(temp1)
    traces=temp1[len1-1]
    head='%!PS-Adobe-2.0 EPSF-1.2\n%%BoundingBox: 0 40 700 740\n%%Pages: 1\n%%EndComments\nnewpath\n'
    con=head+traces
    out.write(con)
    out.flush()
    out.close()

  @staticmethod
  def get_traces(file):
    s=open(file)
    filedir=file.split('.eps')
    filename=filedir[0].split('/')
    name=filename[len(filename)-1]+'.eps'
   # print(name)

    con=s.read()

    temp1=con.split('fill\nstroke\nnewpath\n')
    len1=len(temp1)
    traces=temp1[len1-1]
    return traces,name

  @staticmethod
  def extract_traces(traces,outdir,fn,box):
    traces_split=traces.split('\nnewpath\n')
    L=len(traces_split)
    subi=0
    fn=fn.split('.eps')[0]
    create_newdir(outdir+'/'+fn)
    for sub_trace in traces_split:
      coords_split=sub_trace.split('moveto\n')
      first=coords_split[0].split(' ')
      lines=coords_split[1]
     # print(lines)
      rest=lines.split('lineto\n')
      LR=len(rest)
      coords=np.zeros([LR,2])
      coords[0,0]=float(first[0])
      coords[0,1]=float(first[1])
      for i in range(1,LR):
        coords[i,0]=float(rest[i-1].split(' ')[0])
        coords[i,1]=float(rest[i-1].split(' ')[1])
      coords[:,0]-=np.min(coords[:,0])
      coords[:,1]-=np.min(coords[:,1])
      sizex,sizey=int(np.max(coords[:,0]))+1,int(np.max(coords[:,1]))+1
     # head='%!PS-Adobe-2.0 EPSF-1.2\n%%BoundingBox: 0 0 '+str(box)+' '+str(box)+'\n%%Pages: 1\n%%EndComments\nnewpath\n'
      head='%!PS-Adobe-2.0 EPSF-1.2\n%%BoundingBox: 0 0 '+str(sizex)+' '+str(sizey)+'\n%%Pages: 1\n%%EndComments\nnewpath\n'

      subi+=1
      

      sub_out=outdir+'/'+fn+'/sub_'+str(subi)+'.eps'
      print(sub_out)
      sub_file=open(sub_out,'w')
      sub_file.write(head)
      sub_file.write(str(coords[0,0]))
      sub_file.write(' ')
      sub_file.write(str(coords[0,1]))       
      sub_file.write(' moveto\n')

      for i in range(1,LR):
        sub_file.write(str(coords[i,0]))
        sub_file.write(' ')
        sub_file.write(str(coords[i,1]))       
        sub_file.write(' lineto\n')
      tail='0.5 setlinewidth\n0 0 0 setrgbcolor\nstroke'
      sub_file.write(tail)
      sub_file.close()
    
  