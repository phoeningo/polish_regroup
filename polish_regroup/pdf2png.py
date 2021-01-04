import fitz
import time
import re
import os
import argparse

parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input',type=str)
parser.add_argument('--output',type=str)
args=parser.parse_args()


def pdf2pic(path, pic_path):
    doc=fitz.open(path)
    Len=len(doc)   
    for i in range(Len):
        pic=doc[i].getPixmap()
        picname='pic'+str(i)+'.png'
        print(picname)
        pic.writePNG(picname)  
    
if __name__=='__main__':
    path = args.input+'.pdf'
    pic_path = args.output
    if os.path.exists(pic_path):
        print('exists!')
       # pass
    else:
        os.mkdir(pic_path)
    pdf2pic(path, pic_path)
