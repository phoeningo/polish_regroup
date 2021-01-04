import cv2
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

i=0
final_pic=0
train_filelist=glob('RAND/*.png')
pca=PCA(0.99)
for files in train_filelist:
  proj=EX.rect(files)
  #proj=pca.fit_transform(proj)
  #print(proj.shape)
  x,y=proj.shape
  proj=proj.flatten().reshape(1,x*y)
  if i==0:
    final_pic=proj
  else:
    final_pic=np.concatenate((final_pic,proj))
  i=i+1
  print(i)
  print(final_pic.shape)
#final_pic=pca.fit_transform(final_pic)
#print(final_pic.shape)
  
X=final_pic
#==============================
#kmeans = KMeans(n_clusters=10)
#kmeans.fit(final_pic)
for index, gamma in enumerate((0.001,0.002,0.003,0.0001)):
  for index, k in enumerate((4,5)):
    y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
    print ("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:", metrics.calinski_harabaz_score(X, y_pred))

#preds=SpectralClustering().fit_predict(final_pic)
#print(metrics.calinski_harabaz_score(final_pic, preds))
#labels=kmeans.predict(final_pic)
#print(preds)
'''
L=len(train_filelist)
for i in range(L):
  cmd='cp '+train_filelist[i]+' '+str(preds[i])+'/'
  print(cmd)
  os.system(cmd)

'''
#sc=SpectralClustering(affinity='nearest_neighbors',n_clusters=6,n_neighbors=10)
#sc.fit(final_pic)
#for i in range(1,100):
#  db=DBSCAN(eps=float(i)/10,min_samples=3).fit(final_pic)
#  labels=db.labels_
#  print(labels)

#======================================
#print(cmd)
#os.system(cmd)

'''
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 
#for x in range(i):
#  print(labels[x])
for xi in range(n_clusters_):
    one_cluster = final_pic[labels == i]
    print(one_cluster)
    plt.plot(one_cluster[:,0],one_cluster[:,1],'o')
plt.show()


final_pic=0

i=0

filelist=glob('ALLPIC/*.png')
for files in filelist:
  i=i+1
  print(i)
  pic=cv2.imread(files)
  g_p=cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY)
  pic_array=np.float32(g_p)
  #tsne = TSNE(n_components=2, init='pca', random_state=0)
  #proj = tsne.fit_transform(pic_array)
  proj=255-pic_array
  x,y=proj.shape
  proj=proj.flatten().reshape(1,x*y)
  cmd='cp '+files+ ' SC/'+str(int(sc.fit_predict(proj)))+'/'
  print(cmd)
  os.system(cmd)
#  print(files+'is cluster%i'%(int(kmeans.predict(pic))))
  
'''
   

