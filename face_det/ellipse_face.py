import cv2
import numpy as np
import matplotlib.pyplot as plt

x, y, ra, rb, theta = (169.8556976, 214.7783051, 142.5768585, 101.1835785, -1.508729219)
x = int(np.round(x))
y = int(np.round(y))
ra = int(np.round(ra))
rb = int(np.round(rb))
angle = theta * 180 / np.pi

img = cv2.imread('./samples/image00168.jpg')

# 椭圆的坐标只能是整数
cv2.ellipse(img, center=(x, y), axes=(ra, rb), angle=angle, startAngle=0, endAngle=360, color=(255, 255, 255))

# 根据椭圆得到矩形框
cv2.circle(img, (x,y), 2, (0, 255, 0), 2) # 椭圆中心

# 椭圆长直径
rx1 = int(np.round(ra * np.cos(theta)))
ry1 = int(np.round(ra * np.sin(theta)))
cv2.line(img, (x, y), (x+rx1, y+ry1), (255, 255, 255), 1)
cv2.line(img, (x, y), (x-rx1, y-ry1), (255, 255, 255), 1)

# 椭圆短直径
rx2 = int(np.round(rb * np.sin(-theta)))
ry2 = int(np.round(rb * np.cos(-theta)))
cv2.line(img, (x,y), (x+rx2, y+ry2), (255, 255, 255), 1)
cv2.line(img, (x,y), (x-rx2, y-ry2), (255, 255, 255), 1)


# 椭圆外接矩形
theta2 = np.arctan(rb/ra * np.tan(theta))
w = ra * np.cos(theta2)*np.cos(theta) + rb * np.sin(theta2)*np.sin(theta)

theta3 = np.arctan(-rb / (ra * np.tan(theta)))
h = -ra * np.cos(theta3)*np.sin(theta) + rb*np.sin(theta3)*np.cos(theta)

w, h = int(np.round(w)), int(np.round(h))
cv2.rectangle(img, (x-w, y-h), (x+w, y+h), (255, 255, 255), 1)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



