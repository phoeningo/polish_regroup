from ex import op as EX
import cv2
import argparse

parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input',type=str)
parser.add_argument('--output',type=str)
args=parser.parse_args()

img=EX.rect(args.input)
cv2.imwrite(args.output,img)