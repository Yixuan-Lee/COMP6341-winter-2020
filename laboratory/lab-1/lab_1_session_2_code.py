import cv2
import numpy as np

# path
path=''

# flag: 0 grayscale
# flag: 1 color
Image=cv2.imread('crayons.jpg',1)


#show image
cv2.imshow('W',Image)
cv2.waitKey(0)
cv2.destroyAllWindows()



#write image
cv2.imwrite('messigray.png',Image)



# simple smoothing
kernel = np.ones((7,7),np.float32)/25


# gaussian smoothing
# guassian kernel
kernel = np.zeros((7,7),np.float32)
varx=2
vary=2
centerx=round(kernel.shape[0]/2)-1
centery=round(kernel.shape[1]/2)-1

for i in range(kernel.shape[0]):
    for j in range(kernel.shape[1]):
        kernel[i,j]=np.exp( -( (i-centerx)**2 /(2*(varx**2)) + (j-centery)**2 / (2*(vary**2)) ) )


kernel= kernel / np.sum(kernel)
dst_Image = cv2.filter2D(Image,-1,kernel)

cv2.imshow('W1',dst_Image)
cv2.waitKey(0)
# cv2.destroyAllWindows()



#writing our own image filtering
#naive way

DstimageB=np.zeros((Image.shape[0],Image.shape[1]),np.float32)
DstimageG=np.zeros((Image.shape[0],Image.shape[1]),np.float32)
DstimageR=np.zeros((Image.shape[0],Image.shape[1]),np.float32)
for i in range(centerx,Image.shape[0]-centerx):
    print(i)
    for j in range(centery,Image.shape[1]-centery):

        val=0
        for k in range (-centerx,centerx):
            for l in range(-centery,centery):
                val= val + Image[i+k,j+l,:] * kernel[ k+ centerx, l+centery ]

        DstimageB[i, j]= val[0]
        DstimageG[i, j] = val[1]
        DstimageR[i, j] = val[2]
Dstimage=np.dstack((DstimageB,DstimageG,DstimageR)).astype(np.uint8)
# Dstimage.astype(np.uint8)
cv2.imshow('W2',Dstimage)
cv2.waitKey(0)
cv2.destroyAllWindows()



# we can do with 2 fors

DstimageB=np.zeros((Image.shape[0],Image.shape[1]),np.float32)
DstimageG=np.zeros((Image.shape[0],Image.shape[1]),np.float32)
DstimageR=np.zeros((Image.shape[0],Image.shape[1]),np.float32)

# kernelF=np.dstack((kernel,kernel,kernel))

for i in range(centerx,Image.shape[0]-centerx):
    print(i)
    for j in range(centery,Image.shape[1]-centery):

        DstimageB[i, j]= np.sum(Image[i-centerx:i+centerx+1,j-centery:j+centery+1,0] * kernel)
        DstimageG[i, j] = np.sum(Image[i-centerx:i+centerx+1,j-centery:j+centery+1,1] * kernel)
        DstimageR[i, j] = np.sum(Image[i-centerx:i+centerx+1,j-centery:j+centery+1,2] * kernel)


Dstimage=np.dstack((DstimageB,DstimageG,DstimageR)).astype(np.uint8)
# Dstimage.astype(np.uint8)
cv2.imshow('W3',Dstimage)
cv2.waitKey(0)
cv2.destroyAllWindows()