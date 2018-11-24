from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import random


all_filepaths_labels = [(dir + '/' + filepath, label) for dir, label in [('images/apples', 0), ('images/bananas', 1)] for filepath in os.listdir(
    dir)]
all_filepaths_labels = list(set(all_filepaths_labels))
# all_filepaths_labels = all_filepaths_labels[:50]  # decrease accuracy
random.shuffle(all_filepaths_labels)
all_images_labels = []
for path, label in all_filepaths_labels:
    image = np.array(Image.open(path).resize((25, 25)))
    image = image.transpose(2, 0, 1)
    image = image.reshape(
        1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
    all_images_labels.append((image / 255., label))

s = len(all_images_labels) - 10
study_images_labels = all_images_labels[:s]
test_images_labels = all_images_labels[s:]

study_images, study_labels = list(zip(*study_images_labels))
test_images, test_labels = list(zip(*test_images_labels))
image_list = np.array(study_images)
Y = to_categorical(study_labels)
# 0 -> [1,0], 1 -> [0,1]

model = Sequential()
model.add(Dense(200, input_dim=1875))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))

opt = Adam(lr=0.001)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])
model.fit(image_list, Y, nb_epoch=100, batch_size=100, validation_split=0.1)


results = model.predict_classes(np.array(test_images))
for answer, result in zip(test_labels, results):
    print("answer:", answer, "result:", result)

print('total', sum(int(answer == result)
                   for answer, result in zip(test_labels, results)) / len(results))
