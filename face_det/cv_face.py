import os
import cv2
from functools import partial
from scipy.misc import imresize
from PIL import Image

img_path = r'./samples/multipie3.png'
image = cv2.imread(img_path)
# image = cv2.flip(image, 1) # 侧脸只能检测其中一个右脸，左脸需要水平翻转

# 检测人脸需要的模型，需要从github的下载.所有xml模型都在opencv/data文件夹下
harr_model_path = './data/haarcascades'
frontal_model= os.path.join(harr_model_path, 'haarcascade_frontalface_default.xml')
profile_model = os.path.join(harr_model_path, 'haarcascade_profileface.xml')

# 正脸检测的模型
frontal_dector = partial(cv2.CascadeClassifier(frontal_model).detectMultiScale,
                         scaleFactor=1.1,
                         minNeighbors=5,
                         minSize=(100, 100))

# 侧脸检测的模型
profile_dector = partial(cv2.CascadeClassifier(profile_model).detectMultiScale,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(100, 100))


# 检测得到的结果
faces = frontal_dector(image)

# 显示结果
print('Number of faces detected : {}'.format(len(faces)))
for (x, y, z, w) in faces:
    cv2.rectangle(image, (x,y), (x+z, y+w), (0, 255, 0), 1)
    save_img = image[y:y+w,x:x+z]
    #save_img = imresize(save_img,[100,100],'bilinear')
    cv2.imwrite('2.jpg', save_img)
    save_img = Image.open('2.jpg')
    save_img = save_img.resize((100,100))
    save_img = save_img.convert('L')
    save_img.save('2.jpg')
    
cv2.imshow('faces', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
