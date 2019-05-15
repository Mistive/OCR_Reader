


# match filter : https://nanamare.tistory.com/18
# matchTemplate 사용법 : https://webnautes.tistory.com/1004
#Feature Detection 알고리즘 설명 : https://laonple.blog.me/220906235537
#Featue Detection 함수 설명 : https://m.blog.naver.com/samsjang/220657746860

import numpy as np
import cv2 as cv

img1 = cv.imread('card_image/04261441301390_cardBox.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('alpabet/name.png',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv.imshow("img3", img3)
cv.waitKey(0)


'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def getROI(source, sh, eh, sw, ew):
   dest =  source[sh:eh, sw, ew]
   return dest


if __name__ == "__main__":
   #image = cv2.imread('04261417320855_cardBox.png', cv2.IMREAD_COLOR)
   img1 = cv2.imread('K.png', cv2.IMREAD_GRAYSCALE)
   img2 = cv2.imread('card.png', cv2.IMREAD_GRAYSCALE)
   #img1 = cv2.resize(img1, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
   #img2 = cv2.resize(img2, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

   orb = cv2.ORB_create()

   kp1, des1 = orb.detectAndCompute(img1, None)
   kp2, des2 = orb.detectAndCompute(img2, None)
   bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

   matches = bf.match(des1, des2)
   matches = sorted(matches, key=lambda x: x.distance)

   img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
   #cv2.imshow("IMG1", img1)
   #cv2.imshow("IMG2", img2)
   cv2.imshow("Feature Matching", img3)
   cv2.waitKey(0)



import numpy as np
import cv2

img1 =cv2.imread("card_image/04261441301390_cardBox.png",cv2.IMREAD_GRAYSCALE)
img2 =cv2.imread("alpabet/K.png",cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
res = None

orb=cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf= cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches=bf.match(des1,des2)

matches = sorted(matches, key=lambda x:x.distance)
res=cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],res,flags=0)

cv2.imshow("Feature Matching",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''