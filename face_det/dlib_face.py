import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()

window = dlib.image_window()
img = io.imread('samples/sample2.jpg')

dets = detector(img, 1)
print('Number of faces detected : {}'.format(len(dets)))

for i, d in enumerate(dets):
    print('Dectection %d: (%f, %f), (%f, %f).' % (i, d.left(), d.top(), d.right(), d.bottom()))

window.clear_overlay()
window.set_image(img)
window.add_overlay(dets)
dlib.hit_enter_to_continue()
