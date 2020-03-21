from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Flatten,MaxPooling2D,Concatenate,concatenate
from keras.models import Sequential
import numpy as np
from keras import optimizers
from readData import *
import os 
import cv2 as cv2
import selectivesearch
import skimage.data
from PIL import Image
import matplotlib
# import matplotlib.pyplot as plt
# def findIOU(boxA, boxB):
# 	# determine the (x, y)-coordinates of the intersection rectangle
# 	xA = max(boxA[0], boxB[0])
# 	yA = max(boxA[1], boxB[1])
# 	xB = min(boxA[2], boxB[2])
# 	yB = min(boxA[3], boxB[3])
# 	# compute the area of intersection rectangle
# 	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
# 	# compute the area of both the prediction and ground-truth
# 	# rectangles
# 	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
# 	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
# 	# compute the intersection over union by taking the intersection
# 	# area and dividing it by the sum of prediction + ground-truth
# 	# areas - the interesection area
# 	iou = interArea / float(boxAArea + boxBArea - interArea)
# 	# return the intersection over union value
# 	return iou


# from keras.preprocesing import image
# img=image.load_img("data/index.jpeg")
# img_arr=np.array(img_to_array(img))
# np.resize(img_arr,(227,227,3))
# image.save_img("data/index_re",)
def drawImage(imgArray):
	# fig =plt.figure()
	img=Image.fromarray(imgArray.astype('uint8'),'RGB')
	img.show('img',img)
	# raw_input()
	# matplotlib.pyplot.pause(10)

def nn():
	model = Sequential()
	inputs= Input(shape=(227,227,3))
	#sfl_input=Flatten(inputs)
	conv1=Conv2D(96,(11,11),strides=(4,4),data_format="channels_last",padding='same')(inputs)
	max1 =MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv1)
	conv2=Conv2D(256,(5,5),strides=(1,1),padding='same')(max1)
	max2 =MaxPooling2D(pool_size=(3, 3), strides=(2,2),padding='same')(conv2)
	conv3=Conv2D(384,(3,3),strides=(1,1),padding='same')(max2)
	conv4=Conv2D(384,(3,3),strides=(1,1),padding='same')(conv3)
	conv5=Conv2D(256,(3,3),strides=(1,1),padding='same')(conv4)

	conv1a=Conv2D(256,(4,4),strides=(4,4),padding='same')(max1)
	conv3a=Conv2D(256,(2,2),strides=(2,2),padding='same')(conv3)
	pool5 =Conv2D(256,(3,3),strides=(2,2),padding='same')(conv5)




	concat3=concatenate([conv1a,conv3a,pool5])
	convall=Conv2D(192,(7,7),padding='valid')(concat3)
	convall=Flatten()(convall)

	fc_full=Dense((3072),activation='linear')(convall)

	fcdet=Dense(572,activation='linear')(fc_full)
	fcdet_fi=Dense(2,activation='softmax',name='detection')(fcdet)

	fcland=Dense(572,activation='linear')(fc_full)
	fcland_fi=Dense(42,activation='linear',name='landmark')(fcland)

	fcvisi=Dense(572,activation='linear')(fc_full)
	fcvisi_fi=Dense(21,activation='linear',name='visibility')(fcvisi)

	fcpose=Dense(572,activation='linear')(fc_full)
	fcpose_fi=Dense(3,activation='linear',name='pose')(fcpose)

	fcgender=Dense(572,activation='linear')(fc_full)
	fcgender_fi=Dense(2,activation='linear',name="gender")(fcgender)

	# categoryLB = LabelBinarizer()
	# colorLB = LabelBinarizer()
	# categoryLabels = categoryLB.fit_transform(categoryLabels)
	# colorLabels = colorLB.fit_transform(colorLabels)

	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	out=[fcdet_fi,fcland_fi,fcvisi_fi,fcpose_fi,fcgender_fi]
	hyper_face=Model(inputs=inputs,outputs=out)
	hyper_face.summary()

	# to_loss={"fcdet_fi":"categorical_crossentropy",
	# 			"fcland_fi":"mean_squared_error",
	# 			"fcvisi_fi":"mean_squared_error",
	# 			"pose":"mean_squared_error",
	# 			"gender":"categorical_crossentropy"
	# }

	# loss_weights={	'fcdet_fi': 1.0,
	#                 'fcland_fi':1.0,
	#                 'fcvisi_fi': 1.0,
	# 				'pose': 1.0,
	#                 'gender': 1.0

	#                 }
	# metrics={'fcdet_fi': 'accuracy', 'fcland_fi':'accuracy','fcvisi_fi': 'accuracy', 'pose':'accuracy','gender':'accuracy'}
	hyper_face.compile(optimizer='adam',loss={"detection":"mean_squared_error",
				"landmark":"mean_squared_error",
				"visibility":"mean_squared_error",
				"pose":"mean_squared_error",
				"gender":"mean_squared_error"
	},loss_weights={	'detection': 1.0,
	                'landmark':1.0,
	                'visibility': 1.0,
					'pose': 1.0,
	                'gender': 1.0

	                }, metrics={'detection': 'accuracy', 'landmark':'accuracy','visibility': 'accuracy', 'pose':'accuracy','gender':'accuracy'}
	)

	return hyper_face
# img=cv2.imread("img1.jpg");
# img = skimage.data.astronaut()
# img=cv2.resize(img,(512,512))
# img_lbl, regions = selectivesearch.selective_search(img, scale=5000, sigma=0.9, min_size=10)
# # print(regions[:10])
# for rectdicts in regions[:10]:
# 	label=rectdicts["labels"]
# 	rect=rectdicts["rect"]
# 	x1=rect[0]
# 	y1=rect[1]
# 	x2=rect[2]
# 	y2=rect[3]
# 	cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
# img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print("img size:",img.shape)
# cv2.imshow("window",img)
# cv2.waitKey(0)
# x_train= np.empty((10,227,227,3))
# train_det=np.ones((10,2)) #[0,1] not face
# train_land=np.empty((10,42))
# train_visi=np.empty((10,21))
# train_pose=np.empty((10,3))
# train_gender=np.empty((10,2))
def train(hyper_face):
	trainList=readFile()
	x_train=(np.asarray(trainList[0])).astype('float32')
	# for img in x_train:
	# 	drawImage(img)
	# plt.show()
	x_train/=255
	train_det=(np.asarray(np.asarray(trainList[1]))).astype('float32')#np.empty((1,2))
	train_land=(np.asarray(trainList[2])).astype('float32')#np.empty((1,42))
	train_visi=(np.asarray(trainList[3])).astype('float32')#np.empty((1,21))
	train_pose=(np.asarray(trainList[4])).astype('float32')#np.empty((1,3))
	train_gender=(np.asarray(trainList[5])).astype('float32')#np.empty((1,2))
	# print(train_land[0])
	# print(train_visi[0])
	# print(train_pose[0])
	# print(train_gender)
	# x_train[0][0][1][2]=np.nan

	if(np.isnan(x_train).any() or np.isnan(train_det).any() or (np.isnan(train_land)).any() or (np.isnan(train_visi)).any() or (np.isnan(train_pose)).any() or (np.isnan(train_gender)).any()):
		print("data incorrect")
	hyper_face.fit(x_train,{'detection':train_det, 'landmark':train_land,'visibility':train_visi,'pose':train_pose,'gender':train_gender},epochs=4,batch_size=16);
	return 0
def main():
	hyper_face=nn()
	train(hyper_face)
if(__name__=="__main__"):
	main()
