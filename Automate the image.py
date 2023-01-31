#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os
os.getcwd()


# In[2]:


image1 = Image.open("bridge.jpg")

image1


# In[3]:


image1.show()


# In[4]:


#Changing File type (Extension)
image1.save("bridge.png")


# In[5]:


# lists all the files and folders in the current working directory
os.listdir()


# In[6]:


for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        print(fn, "&", fext)


# In[7]:


#Looping over the image files
for f in os.listdir("."):
    if f.endswith(".jpg"):
        print(f)


# In[8]:


# Creating new Directory using OS library
os.mkdir('NewExtnsn')
# Note: If you already have a directory with this name, you will get error.
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        i.save("NewExtnsn/{}.pdf".format(fn))
        #look into NewExtnsn directory


# In[9]:


# Creating new multiple Directories using OS library
os.makedirs('resize//small')
os.makedirs('resize//tiny')
# Note: If you already have a directory with this name, you will get error.
size_small = (600,600) # small images of 600 X 600 pixels
size_tiny = (200,200)  # tiny images of 200 X 200 pixels
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        i.thumbnail(size_small)
        i.save("resize/small/{}_small{}".format(fn, fext))
        i.thumbnail(size_tiny)
        i.save("resize/tiny/{}_tiny{}".format(fn, fext))
        #look into resize//small and resize//tiny directory


# In[10]:


#Converting to Black and White
image2 = Image.open("bridge.jpg")
image2 = image2.convert(mode='L')
image2


# In[21]:


# Creating new Directory using OS library
os.mkdir('b&w')
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        im = i.convert(mode = 'L')
        im.save("b&w/{}_bw.{}".format(fn, fext))
        #look into b&w directory


# In[12]:


#Rotating the Images
#rotating the image to 55 Degree angle
image3 = Image.open("flower3.jpg")
image3.rotate(55).save("image3.jpg")
Image3 = Image.open("image3.jpg")
Image3


# In[20]:


# Creating new Directory using OS library
os.mkdir('rotate')
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        im = i.rotate(90)
        im.save("rotate/{}_rot.{}".format(fn, fext))
        #look into rotate directory


# In[ ]:




