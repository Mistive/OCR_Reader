import cv2
from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])
Rect = namedtuple("Rectangle", ['p1', 'p2','cp', 'w', 'h','area'])


def getRectangle(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    cp = Point(x + int(w / 2), y + int(h / 2))
    rect = Rect(Point(x, y), Point(x + w, y + h), cp, w, h, w * h)
    return rect




if __name__ == "__main__":
    img_color = cv2.imread('img2.jpg', cv2.IMREAD_COLOR)
    img_Canny = cv2.Canny(img_color, 150, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_Morph = cv2.morphologyEx(img_Canny, cv2.MORPH_DILATE, kernel);
   #img_Morph = cv2.morphologyEx(img_Morph, cv2.MORPH_CLOSE, kernel);

    contours, hierachy = cv2.findContours(img_Canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 70 < area:
            rect = getRectangle(cnt)
            cv2.rectangle(img_color, rect.p1, rect.p2, (255,0,0), 2)

    cv2.imshow('image', img_color)
    cv2.imshow('Canny', img_Canny)
    cv2.imshow('Morph', img_Morph)


    cv2.waitKey(0)


