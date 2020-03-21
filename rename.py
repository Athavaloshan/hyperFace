import os 
  
# Function to rename multiple files 
 
i = 0
  
for filename in os.listdir("/home/athavaloshan/Downloads/NitroShare/face_detection/aflw/aflw-images-3/aflw/data/flickr/4/"): 
    dsta=filename.split('-')
    dst=dsta[0]+'.jpg'
    print(filename,":",dst,"\n")
    # rename() function will 
    # rename all the files 
    # os.rename("/home/athavaloshan/Downloads/NitroShare/face_detection/aflw/aflw-images-0/aflw/data/flickr/0/"+filename, "/home/athavaloshan/Downloads/NitroShare/face_detection/aflw/aflw-images-0/aflw/data/flickr/0/"+dst) 
    i += 1