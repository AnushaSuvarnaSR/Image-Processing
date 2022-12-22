#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

img1 = cv2.imread("flower.jpg")
img2 = cv2.imread("flower3.jpg")
bitwise_and = cv2.bitwise_and(img2, img1)

cv2.imshow("bit_and", bitwise_and)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2

img1 = cv2.imread("flower.jpg")
img2 = cv2.imread("flower3.jpg")
bitwise_or = cv2.bitwise_or(img2, img1)

cv2.imshow("bit_or", bitwise_or)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2

img2 = cv2.imread("flower.jpg")
bitwise_not = cv2.bitwise_not(img2)

cv2.imshow("bitwise_not", bitwise_not)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2

img1 = cv2.imread("flower.jpg")
img2 = cv2.imread("flower3.jpg")
bitwise_xor = cv2.bitwise_xor(img2, img1)

cv2.imshow("bit_xor", bitwise_xor)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2

img1 = cv2.imread("flower.jpg")
img2 = cv2.imread("flower3.jpg")
bitwise_or = cv2.bitwise_or(img2, img1)

cv2.imshow("bit_or", bitwise_or)

bitwise_xor = cv2.bitwise_xor(img2, img1)

cv2.imshow("bit_xor", bitwise_xor)

bitwise_and = cv2.bitwise_and(img2, img1)

cv2.imshow("bit_and", bitwise_and)

bitwise_not = cv2.bitwise_not(img2)

cv2.imshow("bitwise_not", bitwise_not)


cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:




