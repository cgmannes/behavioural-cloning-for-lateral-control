import csv
import cv2
import random
import numpy as np  
import matplotlib.image as img  
import matplotlib.pyplot as plt   
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean

# Locations of folders in directory.
logName = '/Volumes/SEAGATE2/me780ProjectCode/data/driving_log.csv'
dirName = '/Volumes/SEAGATE2/me780ProjectCode/data'

logName2 = '/Volumes/SEAGATE2/me780ProjectCode/straight1/driving_log.csv'
dirName2 = '/Volumes/SEAGATE2/me780ProjectCode/straight1'

logName3 = '/Volumes/SEAGATE2/me780ProjectCode/straight2/driving_log.csv'
dirName3 = '/Volumes/SEAGATE2/me780ProjectCode/straight2'

logName4 = '/Volumes/SEAGATE2/me780ProjectCode/udacitySaif/driving_log.csv'
dirName4 = '/Volumes/SEAGATE2/me780ProjectCode/udacitySaif'


def plot_steeringHist(steeringAngle, title):

    fig = plt.figure()

    plt.hist(steeringAngle, bins=100, color='blue', linewidth=0.5)

    plt.title('Steering Angle Histogram', fontsize=25)
    plt.xlabel('Steering Angle', size=20)
    plt.ylabel('Counts', size=20)

    plt.show()

    fig.savefig(title + '.png', bbox_inches='tight')

    return


def dataAddition(name, sAngle, imgSet, steering, counter):
    
    iter1 = range(1)
    iter2 = range(3)
    counter += 1

    if sAngle > 0 and sAngle <= 0.2 or sAngle < 0 and sAngle >= -0.2:
        imgSet.append(name)
        steering.append(sAngle)
        for i in iter1:
            imgSet.append(name)
            noise = random.uniform(-0.05,0.05)
            noiseyAngle = sAngle + noise
            steering.append(noiseyAngle)

    elif sAngle > 0.2 and sAngle < 0.6 or sAngle < -0.2 and sAngle > -0.6:
        imgSet.append(name)
        steering.append(sAngle)
            
        for j in iter2:
            imgSet.append(name)
            noise = random.uniform(-0.05,0.05)
#             noiseyAngle = sAngle + noise

            if (sAngle + noise) > 0.6:
#                 noiseyAngle = 0.6
                steering.append(0.6)
            elif (sAngle + noise) < -0.6:
#                 noiseyAngle = -0.6
                steering.append(-0.6)
            else:
                steering.append(sAngle + noise)
    else:
        counter -=1

    return name, sAngle, imgSet, steering, counter


# Convert CSV to set of variables.
imgSet = []
steering = []

counter = 0
with open(logName) as csv_file:
    csv_log = csv.reader(csv_file, delimiter = ',')
    for row in csv_log:
        if counter == 0:
            counter += 1
            continue
        elif float(row[3]) == 0:
            continue
        elif float(row[3]) != 0:
            name = dirName + '/' + row[0].lstrip()
            sAngle = float(row[3])
            name, sAngle, imgSet, steering, counter = dataAddition(name, sAngle, imgSet, steering, counter)
        else:
            counter += 1
            continue


imgSet2 = []
steering2 = []

counter2 = 0
with open(logName2) as csv_file:
    csv_log = csv.reader(csv_file, delimiter = ',')
    for row in csv_log:
        if counter2 == 0:
            counter2 += 1
            continue
        else:
            if counter2%3 == 0:
                if float(row[3]) == 0:
                    l = list(row[0])
                    del(l[0:33])
                    fileName2 = "".join(l)
                    imgSet2.append(dirName2 + '/' + fileName2)
                    steering2.append(float(row[3]))
                    counter2 += 1


imgSet3 = []
steering3 = []

counter3 = 0
with open(logName3) as csv_file:
    csv_log = csv.reader(csv_file, delimiter = ',')
    for row in csv_log:
        if counter3 == 0:
            counter3 += 1
            continue
        else:
            if counter3%2 == 0:
                if float(row[3]) == 0:
                    l = list(row[0])
                    del(l[0:30])
                    fileName3 = "".join(l)
                    imgSet3.append(dirName3 + '/' + fileName3)
                    steering3.append(float(row[3]))
                    counter3 += 1
            

S = steering + steering2 + steering3
ImgSet = imgSet + imgSet2 + imgSet3


finalImgSet = []
finalSteering = []
for s in range(len(S)):
        finalSteering.append(S[s]/0.6)
        finalImgSet.append(ImgSet[s])
    
plot_steeringHist(finalSteering, "modifiedSteeringHist")
print('Fig1')

imgSet = finalImgSet
steering = finalSteering

# Converting images to numpy arrays.
width, height = 320, 80

# Set steering angles to Y and horizontally invert
# images and steering angles.
Y = steering
Y_flip = [ -1*n for n in Y ]

# Input data X.
X = np.zeros([2*len(Y), height, width, 3], dtype=np.float32)

counter = 0
# Populating matrix X.
for imgName in imgSet:
    if counter%500 == 0:
        print(counter)
#     if counter > endCounter:
#         break
    image = img.imread(imgName, format='jpg')
    #image = rgb2gray(image)

    # Normalize values
    X[counter,:,:,:] = image[60:140,:,:]/255.0
    X[counter + len(Y),:,:,:] = cv2.flip(X[counter,:,:,:],1)

    counter += 1
    
    
# fig = plt.figure()
# crop = X[200,:,:,:]
# fig = plt.figure(figsize = (10,10))
# plt.imshow(crop)
# plt.show()


# Append steering angles and form a nx1 vector.
Yprime = Y + Y_flip

# Y array.
Y = np.array(Yprime)

plot_steeringHist(Y, "completeSteeringHist")
print('Fig2')

# Save imported data.
np.save('/Users/Chris/Desktop/me780AD/me780Project/X.npy', X)
np.save('/Users/Chris/Desktop/me780AD/me780Project/y.npy', Y)

print("Numpy arrays saved")







