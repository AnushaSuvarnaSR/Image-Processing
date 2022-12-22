#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Import cv2, matplotlib, numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Read the original image and know its type
img1 = cv2.imread('flower2.jpg',0)

# Obtain the size of the original image
[m, n] = img1.shape
print('Image Shape:', m, n)

# Show original image
print('Original Image:')
plt.imshow(img1, cmap="gray")


# Down sampling

# Assign a down sampling rate
# Here we are down sampling the
# image by 4
f = 4

# Create a matrix of all zeros for
# downsampled values
img2 = np.zeros((m//f, n//f), dtype=np.int)

# Assign the down sampled values from the original
# image according to the down sampling frequency.
# For example, if the down sampling rate f=2, take
# pixel values from alternate rows and columns
# and assign them in the matrix created above
for i in range(0, m, f):
	for j in range(0, n, f):
		try:

			img2[i//f][j//f] = img1[i][j]
		except IndexError:
			pass


# Show down sampled image
print('Down Sampled Image:')
plt.imshow(img2, cmap="gray")



# In[17]:


# Up sampling

# Create matrix of zeros to store the upsampled image
img3 = np.zeros((m, n), dtype=np.int)
# new size
for i in range(0, m-1, f):
	for j in range(0, n-1, f):
		img3[i, j] = img2[i//f][j//f]

# Nearest neighbour interpolation-Replication
# Replicating rows

for i in range(1, m-(f-1), f):
	for j in range(0, n-(f-1)):
		img3[i:i+(f-1), j] = img3[i-1, j]

# Replicating columns
for i in range(0, m-1):
	for j in range(1, n-1, f):
		img3[i, j:j+(f-1)] = img3[i, j-1]

# Plot the up sampled image
print('Up Sampled Image:')
plt.imshow(img3, cmap="gray")


# In[4]:


from PIL import Image,ImageDraw,ImageFilter
im1=Image.open('bridge.jpg')
im2=Image.open('flower3.jpg')
mask_im=Image.new("L",im2.size,0)
draw=ImageDraw.Draw(mask_im)
draw.ellipse((210,105,410,300),fill=225)
mask_im_blur=mask_im.filter(ImageFilter.GaussianBlur(10))
back_im=im1.copy()
back_im.paste(im2,(0,0),mask_im_blur)
back_im.show()


# In[8]:


#up sampling
import cv2
import matplotlib.pyplot as plt
from PIL import Image
im = Image.open("flower.jpg")
plt.imshow(im)
plt.show()

im = im.resize((im.width*5, im.height*5), Image.NEAREST)
plt.figure(figsize=(10,10))
plt.imshow(im)
plt.show()

#down sampling
im = Image.open("flower.jpg")
im = im.resize((im.width//5, im.height//5))
plt.figure(figsize=(15,10))
plt.imshow(im)
plt.show()


# In[9]:


import matplotlib.pyplot as plt
im = Image.open('flower.jpg')
plt.figure(figsize=(20,30))
num_colors_list = [1 << n for n in range(8,0,-1)]
snr_list = []
i = 1
for num_colors in num_colors_list:
 im1 = im.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
 plt.subplot(4,2,i), plt.imshow(im1), plt.axis('off')
 snr_list.append(signaltonoise(im1, axis=None))
 plt.title('Image with # colors = ' + str(num_colors) + ' SNR = ' +
 str(np.round(snr_list[i-1],3)), size=20)
 i += 1
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()


# In[4]:




# Importing Image module from PIL package
from PIL import Image
import PIL

# creating a image object (main image)
im1 = Image.open(r"flower3.jpg")

# quantize a image
im1 = im1.quantize(256)

# to show specified image
im1.show()


# In[5]:


import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
im = Image.open("flower3.jpg")
plt.figure(figsize=(20,30))
num_colors_list = [1 << n for n in range(8,0,-1)]
snr_list = []
i = 1
for num_colors in num_colors_list:
 im1 = im.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
 plt.subplot(4,2,i), plt.imshow(im1), plt.axis('off')
 snr_list.append(signaltonoise(im1, axis=None))
 plt.title('Image with # colors = ' + str(num_colors) + ' SNR = ' +
 str(np.round(snr_list[i-1],3)), size=20)
 i += 1
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()


# In[6]:


import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
im = Image.open("house.jpg")
plt.figure(figsize=(20,30))
num_colors_list = [1 << n for n in range(8,0,-1)]
snr_list = []
i = 1
for num_colors in num_colors_list:
 im1 = im.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
 plt.subplot(4,2,i), plt.imshow(im1), plt.axis('off')
 snr_list.append(signaltonoise(im1, axis=None))
 plt.title('Image with # colors = ' + str(num_colors) + ' SNR = ' +
 str(np.round(snr_list[i-1],3)), size=20)
 i += 1
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()


# In[ ]:




