#!/usr/bin/env python
# coding: utf-8

# In[36]:


from PIL import Image
img = Image.open('flower2.jpg')
img.save("D:/flower2.tiff",'TIFF')


# In[17]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.camera()  
type(image)
np.ndarray 


# In[18]:


mask = image < 87  
image[mask]=255  
plt.imshow(image, cmap='gray')


# In[1]:


from PIL import Image

def changeImageSize(maxWidth, 
                    maxHeight, 
                    image):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage
    
  
image1 = Image.open("flower.jpg")
image2 = Image.open("flower3.jpg")


image3 = changeImageSize(800, 500, image1)
image4 = changeImageSize(800, 500, image2)


image5 = image3.convert("RGBA")
image6 = image4.convert("RGBA")


image5.show()
image6.show()

alphaBlended1 = Image.blend(image5, image6, alpha=.2)
alphaBlended2 = Image.blend(image5, image6, alpha=.4)

alphaBlended1.show()
alphaBlended2.show()


# In[19]:


from PIL import Image

#Create an Image Object from an Image
im = Image.open('D:/images/cat.jpg')

#Display actual image
im.show()

#left, upper, right, lowe
#Crop
cropped = im.crop((1,2,300,300))

#Display the cropped portion
cropped.show()

#Save the cropped image
cropped.save('d:/images/croppedBeach1.jpg')


# In[21]:


# import Pillow modules

from PIL import Image

from PIL import ImageFilter

 

# Load the image

img = Image.open("D:/images/cat.jpg");

 

# Display the original image

img.show()

 

# Read pixels and apply negative transformation

for i in range(0, img.size[0]-1):

    for j in range(0, img.size[1]-1):

        # Get pixel value at (x,y) position of the image

        pixelColorVals = img.getpixel((i,j));

       

        # Invert color

        redPixel    = 255 - pixelColorVals[0]; # Negate red pixel

        greenPixel  = 255 - pixelColorVals[1]; # Negate green pixel

        bluePixel   = 255 - pixelColorVals[2]; # Negate blue pixel

                   

        # Modify the image with the inverted pixel values

        img.putpixel((i,j),(redPixel, greenPixel, bluePixel));

 

# Display the negative image

img.show();


# In[26]:


import cv2
import numpy as np
# Load the image
img = cv2.imread("D:/images/cat.jpg")
# Check the datatype of the image
print(img.dtype)
# Subtract the img from max value(calculated from dtype)
img_neg = 255 - img
# Show the image
cv2.imshow('negative',img_neg)
cv2.waitKey(0)


# In[35]:


from PIL import Image, ImageOps

im = Image.open('D:/images/flower.jpg')
im_invert = ImageOps.invert(im)
im_invert.save('D:/images/negative.jpg', quality=95)


# In[3]:


# Python3 program to draw line
# shape on solid image
import numpy as np
import cv2

# Creating a black image with 3 channels
# RGB and unsigned int datatype
im = cv2.imread('house.jpg')


# Creating line
cv2.line(im, (20, 160), (100, 160), (0, 0, 255), 10)

cv2.imshow('dark', im)

# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


# Python3 program to draw rectangle
# shape on solid image
import numpy as np
import cv2

# Creating a black image with 3
# channels RGB and unsigned int datatype

img = cv2.imread('house.jpg')
# Creating rectangle
cv2.rectangle(img, (30, 30), (300, 200), (0, 255, 0), 5)

cv2.imshow('dark', img)

# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


# Python3 program to draw circle
# shape on solid image
import numpy as np
import cv2

# Creating a black image with 3
# channels RGB and unsigned int datatype
img = cv2.imread('house.jpg')

# Creating circle
cv2.circle(img, (200, 200), 80, (255, 0, 0), 3)

cv2.imshow('dark', img)

# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


# Python3 program to write
# text on solid image
import numpy as np
import cv2

# Creating a black image with 3
# channels RGB and unsigned int datatype
img = cv2.imread('house.jpg')

# writing text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Hello!!!', (50, 50),
font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('dark', img)

# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


# importing required libraries of opencv
import cv2

# importing library for plotting
from matplotlib import pyplot as plt

# reads an input image
img = cv2.imread('house.jpg',0)

# find frequency of pixels in range 0-255
histr = cv2.calcHist([img],[0],None,[256],[0,256])

# show the plotting graph of an image
plt.plot(histr)
plt.show()


# In[9]:


# importing required libraries of opencv
import cv2

# importing library for plotting
from matplotlib import pyplot as plt

# reads an input image
img = cv2.imread('flower.jpg',0)

# find frequency of pixels in range 0-255
histr = cv2.calcHist([img],[0],None,[256],[0,256])

# show the plotting graph of an image
plt.plot(histr)
plt.show()


# In[10]:


import cv2
from matplotlib import pyplot as plt
img = cv2.imread('house.jpg',0)
 
# alternative way to find histogram of an image
plt.hist(img.ravel(),256,[0,256])
plt.show()


# In[1]:


from PIL import Image,ImageStat
im=Image.open('flower.jpg')
stat=ImageStat.Stat(im)
print(stat.median)


# In[2]:


from PIL import Image,ImageStat
im=Image.open('flower.jpg')
stat=ImageStat.Stat(im)
print(stat.stddev)


# In[8]:


from PIL import Image,ImageStat
im=Image.open('flower.jpg')
stat=ImageStat.Stat(im)
print(stat.mean)


# In[9]:


#RGB Channels
import matplotlib.pyplot as plt
im1=Image.open("flower.jpg")
ch_r,ch_g,ch_b=im1.split()
plt.figure(figsize=(18,6))
plt.subplot(1,3,1);
plt.imshow(ch_r,cmap=plt.cm.Reds);plt.axis('off')
plt.subplot(1,3,2);
plt.imshow(ch_g,cmap=plt.cm.Greens);plt.axis('off')
plt.subplot(1,3,3);
plt.imshow(ch_b,cmap=plt.cm.Blues);plt.axis('off')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




