from keras.models import load_model
import cv2
import numpy
from os import listdir
from os.path import isfile, join
import os

classifier = load_model("monkey.h5")

monkey_breeds_dict = {
    "[0]": "mantled_howler",
    "[1]": "patas_monkey",
    "[2]": "bald_uakari",
    "[3]": "japanese_macaque",
    "[4]": "pygny_marnoset",
    "[5]": "white_headed_capuchin",
    "[6]": "silvery_marmoset",
    "[7]": "common_squirrel_monkey",
    "[8]": "black_headed_night_monkey",
    "[9]": "nilgiri_langur"
}

def draw_test(name, pred, im):
    monkey = monkey_breeds_dict[str(pred)]
    BLACK =[0,0,0]
    expanded_image = cv2.copyMakeBorder(im,80,0,0,100,cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, monkey, (20,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path,x)), os.listdir(path)))
    # print(folders)
    random_directory = numpy.random.randint(0, len(folders))
    path_class = folders[random_directory]
    # print("class - " + monkey_breeds_dict[str(path_class)])
    file_path = path+path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = numpy.random.randint(0, len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)

for i in range(0,10):
    input_im = getRandomImage("./monkey_breed/validation/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx =0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    input_im = cv2.resize(input_im, (224,224), interpolation=cv2.INTER_LINEAR)
    input_im = input_im /255.
    input_im  = input_im.reshape(1,224,224,3)

    res = numpy.argmax(classifier.predict(input_im,1,verbose=0), axis=1)
    draw_test("Prediction", res, input_original)
    cv2.waitKey(0)
cv2.destroyAllWindows()
