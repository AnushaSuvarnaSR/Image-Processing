#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
image=cv2.imread('flower2.jpg')
cv2.imshow('display image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


cv2.imwrite('D:\im.jpg',image)
cv2.waitKey(0)


# In[3]:


new_image=cv2.rotate(image,cv2.ROTATE_180)
cv2.imshow('display image',new_image)
cv2.waitKey(0)


# In[4]:


import cv2
import matplotlib.pyplot as plt
image=plt.imread('flower2.jpg')
plt.imshow(image)
plt.show()


# In[5]:


image.size


# In[6]:


image.shape


# In[7]:


h,w,c=image.shape
print("width:",w)
print("height:",h)
print("channel:",c)


# In[8]:


ret, bw_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
 
# converting to its binary form
bw = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
 
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[12]:



from PIL import Image

im = Image.open("d:\im.jpg")


im.show()

resized_im = im.resize((round(im.size[0]*0.5), round(im.size[1]*0.5)))


resized_im.show()


resized_im.save('resizedBeach1.jpg')


# In[14]:


import cv2


image = cv2.imread('d:im.jpg')
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)

while True:
  
    cv2.imshow('image', image)
    key = cv2.waitKey()

    if key == 27:
        break


cv2.destroyAllWindows()


# In[15]:


down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
cv2.imshow('Resized Down by defining height and width', resized_down)
cv2.waitKey()
cv2.destroyAllWindows()


# In[16]:


up_width = 600
up_height = 400
up_points = (up_width, up_height)
# resize the image
resized_up = cv2.resize(image, up_points, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Resized Up image by defining height and width', resized_up)
cv2.waitKey()
cv2.destroyAllWindows()


# In[17]:


print('The Shape of the image is:',image.shape)
print('The image as array is:')
print(image)


# In[19]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
w, h = 512, 512
data = np. zeros((h, w, 3),dtype=np.uint8)
data[0:300, 0:300] = [255, 0, 255] # red patch in upper left.
img = Image.fromarray(data,'RGB')
img.save('my.png')


# In[20]:


images=plt.imread('my.png')
plt.imshow(images)
plt.show()


# In[4]:


import cv2
image=cv2.imread("flower1.jpg")
color=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
cv2.imshow("display",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


import cv2
image=cv2.imread("flower1.jpg")
color=cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow("display",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


import cv2
image=cv2.imread("flower1.jpg")
color=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
cv2.imshow("display",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


import cv2
image=cv2.imread("flower1.jpg")
color=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
cv2.imshow("display",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


import cv2
image=cv2.imread("flower1.jpg")
color=cv2.cvtColor(image,cv2.COLOR_HLS2RGB)
cv2.imshow("display",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


import cv2
image=cv2.imread("flower1.jpg")
color=cv2.cvtColor(image,cv2.COLOR_LAB2BGR)
cv2.imshow("display",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


import cv2
image=cv2.imread("flower1.jpg")
color=cv2.cvtColor(image,cv2.COLOR_LAB2HLS)
cv2.imshow("display",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
image=cv2.imread("flower1.jpg")
color=cv2.cvtColor(image,cv2.COLOR_LAB2YUV)
cv2.imshow("display",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


import cv2
import matplotlib.pyplot as plt
imge1=plt.imread('flower.jpg')
imge2=plt.imread('flower.jpg')
plt.imshow(imge1)
plt.show()
plt.imshow(imge2)
plt.show()
addimg=imge1+imge2
plt.imshow(addimg)
plt.show()


# In[8]:


import cv2
import matplotlib.pyplot as plt
imge1=plt.imread('flower.jpg')
imge2=plt.imread('flower.jpg')
plt.imshow(imge1)
plt.show()
plt.imshow(imge2)
plt.show()
addimg=imge1-imge2
plt.imshow(addimg)
plt.show()


# In[9]:


import cv2
import matplotlib.pyplot as plt
imge1=plt.imread('flower.jpg')
imge2=plt.imread('flower.jpg')
plt.imshow(imge1)
plt.show()
plt.imshow(imge2)
plt.show()
addimg=imge1*imge2
plt.imshow(addimg)
plt.show()


# In[10]:


import cv2
import matplotlib.pyplot as plt
imge1=plt.imread('flower.jpg')
imge2=plt.imread('flower.jpg')
plt.imshow(imge1)
plt.show()
plt.imshow(imge2)
plt.show()
addimg=imge1/imge2
plt.imshow(addimg)
plt.show()


# In[12]:


import cv2
import matplotlib.pyplot as plt
imge1=plt.imread('flower.jpg')
imge2=plt.imread('flower3.jpg')
plt.imshow(imge1)
plt.show()
plt.imshow(imge2)
plt.show()
addimg=imge1+imge2
plt.imshow(addimg)
plt.show()


# In[ ]:




