import csv
import numpy as np
def readFile():

	filenamedict={}
	faceidList=[]
	with open('/home/athavaloshan/Downloads/NitroShare/face_detection/aflw/Faces-2.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if(line_count==0):
				line_count=1
				continue
			filename=row[1]
			faceid=row[0]
			filenamedict.update({faceid:filename})
			
			faceidList.append(faceid)
	c=0
	n=0

	pointdict={}
	fecordict={}
	with open('/home/athavaloshan/Downloads/NitroShare/face_detection/aflw/FeatureCoords-2.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
	   
		for row in csv_reader:
			if(line_count==0):
				line_count=1
				continue
			if(c==0 and n==0):
				faceidold=row[0]
				fecor=np.zeros(42)
				points=np.zeros(21)
				fid=row[1]
				x=row[2]
				y=row[3]
				points.flat[int(fid)-1]=1
				fecor.flat[2*(int(fid)-1)]=x
				fecor.flat[2*(int(fid)-1)+1]=y
				n=1
				# print("feature coordinate",fecor)
			elif(faceidold==row[0]):

				c=1
				fid=row[1]
				x=row[2]
				y=row[3]
				points.flat[int(fid)-1]=1
				fecor.flat[2*(int(fid)-1)]=x
				fecor.flat[2*(int(fid)-1)+1]=y
			else:
				pointdict.update({faceidold:points})
				fecordict.update({faceidold:fecor})
				c=0
				# n=0
				faceidold=row[0]
				fecor=np.zeros(42)
				points=np.zeros(21)
				fid=row[1]
				x=row[2]
				y=row[3]
				points.flat[int(fid)-1]=1
				fecor.flat[2*(int(fid)-1)]=x
				fecor.flat[2*(int(fid)-1)+1]=y

	# ///may be last element not added to map
	coordict={}
	with open('/home/athavaloshan/Downloads/NitroShare/face_detection/aflw/FaceRect-2.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
	   
		for row in csv_reader:
			if(line_count==0):
				line_count=1
				continue
			coor=np.zeros(4)
			faceid=row[0]
			coor[0]=row[1]
			coor[1]=row[2]
			coor[2]=row[3]
			coor[3]=row[4]

			coordict.update({faceid:coor})
	genderdict={}
	with open('/home/athavaloshan/Downloads/NitroShare/face_detection/aflw/FaceMetaData-2.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
	   
		for row in csv_reader:
			if(line_count==0):
				line_count=1
				continue
			gender=np.zeros(2)
			faceid=row[0]
			if(row[1]=='m'):
				gender[0]=1
			else:
				gender[1]=1
			genderdict.update({faceid:gender})

	posedict={}
	with open('/home/athavaloshan/Downloads/NitroShare/face_detection/aflw/CamPose-2.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
	   
		for row in csv_reader:
			if(line_count==0):
				line_count=1
				continue
			pose=np.zeros(3)
			faceid=row[0]
			pose[0]=row[1]
			pose[1]=row[2]
			pose[2]=row[3]
			posedict.update({faceid:pose})
	trainlist=[]
	for faceid in faceidList:
		traindDict={}
		traindDict.update({'visi':pointdict.get(faceid)})
		traindDict.update({'land':fecordict.get(faceid)})
		traindDict.update({'rect':coordict.get(faceid)})
		traindDict.update({'pose':posedict.get(faceid)})
		traindDict.update({'gender':genderdict.get(faceid)})
		traindDict.update({'faceid':faceid})
		traindDict.update({'img':filenamedict.get(faceid)})
		
		trainlist.append(traindDict)
	return trainlist
# print("printing faceid")
# for faceid in faceidList:
# 	print(faceid)
# 	coor=coordict.get(faceid)
# 	print (coor)

