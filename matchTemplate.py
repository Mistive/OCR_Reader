import cv2
import numpy as np

def getROI(source, sh, eh, sw, ew):
    dest =  source[sh:eh, sw, ew]
    return dest

#match filter : https://nanamare.tistory.com/18
#matchTemplate 사용법 : https://webnautes.tistory.com/1004
if __name__ == "__main__":
    #image = cv2.imread('04261417320855_cardBox.png', cv2.IMREAD_COLOR)
    image = cv2.imread('card_image/04261441301390_cardBox.png', cv2.IMREAD_COLOR)
    tmp = cv2.imread('alpabet/S2.png', cv2.IMREAD_COLOR)

    h, w = tmp.shape[:2]
    #h, w = 36, 25
    result = cv2.matchTemplate(image, tmp, cv2.TM_CCORR)
    cv2.imshow("result", result)
    print(result)
    #loc = np.where(result >= 0.9)
    #for pt in zip(*loc[::-1]):
    #    cv2.rectangle(image, pt, (pt[0] +w, pt[1]+h), (255, 0, 0), 2)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    #top_left = (91, 178)
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image,top_left,bottom_right,(0,0,255), 2)
    cv2.imshow("image", image)

    cv2.waitKey(0)


