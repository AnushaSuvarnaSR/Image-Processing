#!/usr/bin/env python
# coding: utf-8

# In[1]:


#upsampling

import cv2
import matplotlib.pyplot as plt
image=cv2.imread('bridge.jpg')
cv2.imshow('image before pyrUP',image)
image2=cv2.pyrUp(image)
#cv2.imshow('image after pyrUp',image)
cv2.imshow('UpSample',image2)
plt.imshow(image2)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


#downsampling

import cv2
import matplotlib.pyplot as plt
image=cv2.imread('bridge.jpg')
cv2.imshow('image before pyrDown',image)
image2=cv2.pyrDown(image)
#cv2.imshow('image after pyrDown',image)
cv2.imshow('DownSample',image2)
plt.imshow(image2)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#quantization

import cv2
from PIL import Image
image=Image.open('bridge.jpg')
image.show()
img=image.quantize(19)
img.show()


# In[ ]:


#text on image

from PIL import Image
from PIL import ImageDraw
img = Image.open('house.jpg')
I1 = ImageDraw.Draw(img)
font = ImageFont.truetype("arial",50)
I1.text((28, 36),"GOOD MORNING", fill=(255, 0, 0),font=font)
img.show()
img.save("image.png")

