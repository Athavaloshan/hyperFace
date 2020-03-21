import numpy as np 
import csv
import sys

x_train=[]#np.empty((1,227,227,3))
train_det=[]#np.empty((1,2))
train_land=[]#np.empty((1,42))
train_visi=[]#np.empty((1,21))
train_pose=[]#np.empty((1,3))
train_gender=[]#np.empty((1,2))
def decoder(data):
	dataList=data.split(" ")
	list0=[]
	index=0
	for i in range(227):
		list1=[]
		for j in range(227):
			list2=[]
			for k in range(3):
				index=index+k
				list2.append(dataList[index])
			list1.append(list2)
		list0.append(list1)
	x_train.append(list0)
	list0=[]
	for i in range(2):
		list0.append(dataList[index+i])
	index=index+2
	train_det.append(list0)

	list0=[]
	for i in range(42):
		
		# print(index,len(dataList))
		list0.append(dataList[index+i])
	index=index+42
	train_land.append(list0)
	
	list0=[]
	for i in range(21):
		
		list0.append(dataList[index+i])
	index=index+21
	train_visi.append(list0)
	list0=[]
	for i in range(3):
		
		list0.append(dataList[index+i])
	index=index+3
	train_pose.append(list0)
	list0=[]
	for i in range(2):
		
		list0.append(dataList[index+i])
	index=index+2
	train_gender.append(list0)
	return 0

def readFile():
	csv.field_size_limit(sys.maxsize)

	with open('data.csv','r') as csvfile:
		print("start")
		lines=csv.reader(csvfile,delimiter='\n')
		# print(lines)
		for row in lines:
			# print(row)
			decoder(row[0])
	# print(train_land[0])
	# print(train_visi[0])
	# print(train_pose[0])
	# print(train_gender[0])
	trainList=[x_train,train_det,train_land,train_visi,train_pose,train_gender]
	return trainList
