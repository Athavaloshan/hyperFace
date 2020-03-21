from readcsv import *
import numpy as np
import cv2
import selectivesearch
trainDataCount = 2
x_train=[]#np.empty((1,227,227,3))
train_det=[]#np.empty((1,2))
train_land=[]#np.empty((1,42))
train_visi=[]#np.empty((1,21))
train_pose=[]#np.empty((1,3))
train_gender=[]#np.empty((1,2))

def findIOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def populate(datalist):
	count=0
	for data in datalist:
		if(count==trainDataCount):
			break
		count=count+1
		box=data["rect"]
		
		img=data["img"]
		
		img=cv2.imread("/home/athavaloshan/Downloads/NitroShare/face_detection/data/"+img)
	
		img_lbl, regions = selectivesearch.selective_search(img, scale=5000, sigma=0.9, min_size=30)
		
		##0-ignore,1,face,-1 not face
		for regiondict in regions:
			candidatebox=regiondict["rect"]
			
			iou=findIOU(box,candidatebox)
			
			if(iou<3.5):faceIndex=-1
			elif(iou>0.7): faceIndex=1
			else: faceIndex=0
			img1=img[candidatebox[0]:candidatebox[2],candidatebox[1]:candidatebox[3]]
			if(faceIndex!=0 and img1.shape[0]>0 and img1.shape[1]):
				
				x_train.append(cv2.resize(img1,(227,227)))
				
				if(iou==-1): train_det.append(np.array([0,1]))
				else: train_det.append(np.array([1,0]))
				train_land.append(data['land'])
				# print("count:",count,"land:",train_land)

				train_visi.append(data['visi'])
				train_pose.append(data['pose'])
				train_gender.append(data['gender'])
def writeFile(csvlist):
	with open('data.csv', 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter='\n')
		for li in csvlist:
		    spamwriter.writerow([li])
	csvfile.close()
		   
def addtolist(list,n,string):
	for i in range(n):
		string=string+str(list[i])+' '
	return string
	
def loadCSV():
	k=len(x_train)
	count=0
	csvlist=[]
	while(count<k):
		# s=len(train_gender)
		# print(s)
		# (n,m)=x_train[k].shape
		
		str0=''
		for i in range(227):
			for j in range(227):
				str0=addtolist(x_train[count][i][j][:],3,str0)
		str0=addtolist(train_det[count],2,str0)
		str0=addtolist(train_land[count],42,str0)
		
		str0=addtolist(train_visi[count],21,str0)
	
		str0=addtolist(train_pose[count],3,str0)
		
		str0=addtolist(train_gender[count],2,str0)
		csvlist.append(str0)
		# print(x_train[i][:][:][0].shape)
		# print("csvlist",csvlist)
		count=count+1
	return csvlist
def main():
	trainList=readFile()
	# print(trainList[0])
	# print("line")
	# print(trainList[1])
	populate(trainList)
	csvlist=loadCSV();
	writeFile(csvlist)
if(__name__ == "__main__"):
	main()