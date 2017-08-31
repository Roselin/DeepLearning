from conv import Conv2D 
import numpy as np
import cv2 
import time
import matplotlib.pyplot as plt


img = cv2.imread('Image1.jpg')
height, width, depth = img.shape
print ("ImageA height: " + str(height)) 
print ("ImageA width: " + str(width))
'''
#PartA
#Task 1
print ("Start PartA Task1")
conv2d = Conv2D(3,1,3,1,'known') #(in_channel, o_channel, kernel_size, stride, mode)
(Atask1Count, Img1) = conv2d.forward(img)
imTest = Img1[:,:,0]
cv2.imwrite("Task1_k1.jpg", Img1[:,:,0])
#v2.imshow('image',Img1[:,:,0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#Task2
print ("Start PartA Task2")
conv2d = Conv2D(3,2,5,1,'known') #(in_channel, o_channel, kernel_size, stride, mode)
(Atask2Count, Img2) = conv2d.forward(img)
cv2.imwrite("Task2_k4.jpg", Img2[:,:,0])
cv2.imwrite("Task2_k5.jpg", Img2[:,:,1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Task3
print ("Start PartA Task3")
conv2d = Conv2D(3,3,3,2,'known') #(in_channel, o_channel, kernel_size, stride, mode)
(Atask3Count, Img3) = conv2d.forward(img)
cv2.imwrite("Task3_k1.jpg", Img3[:,:,0])
cv2.imwrite("Task3_k2.jpg", Img3[:,:,1])
cv2.imwrite("Task3_k3.jpg", Img3[:,:,2])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
'''

#PartB
print ("Start PartB")
elapsed_time = []
xAxis = []
for i in range(11):
    print ("number of channel: " + str(i))
    xAxis.append(i)
    conv2d = Conv2D(3,i,3,1,'rand') #(in_channel, o_channel, kernel_size, stride, mode)
    start_time = time.time()
    (Btask1Count, iB) = conv2d.forward(img)
    elapsed_time.append(time.time()-start_time)

plt.plot(xAxis, elapsed_time,'ro',xAxis, elapsed_time,'r-' )
plt.xlabel('number of o_channel')
plt.ylabel('Consumming Time (s)')
plt.show()
plt.savefig('PartB.jpg')


#partC
print ("Start PartC")
xAxis = []
yAxis = []
for i in range(3,12,2):
    print ("number of kernal: "+ str(i))
    xAxis.append(i)
    conv2d = Conv2D(3,2,i,1,'rand') #(in_channel, o_channel, kernel_size, stride, mode)
    (count, iC) = conv2d.forward(img)
    yAxis.append(count)

plt.plot(xAxis, yAxis,'ro',xAxis, yAxis,'r-' )
plt.xlabel('Kernal Size')
plt.ylabel('Number of Step')
plt.show()
plt.savefig('PartC.jpg')


