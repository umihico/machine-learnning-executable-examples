from keras.models import Sequential
from keras.layers import Activation, Dense
import numpy as np

import random

input_X_study = [(random.randint(0, 1), random.randint(0, 1))
                 for r in range(10000)]
'''
$ print(input_X_study[:10])
>> [[0, 0], [0, 0], [0, 0], [1, 0], [1, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]
'''
answer_dict = {
    (1, 0): 1,
    (0, 1): 1,
    (1, 1): 0,
    (0, 0): 0,
}  # xnor

output_Y_study = [[answer_dict[x], ] for x in input_X_study]
X = np.array(input_X_study)
Y = np.array(output_Y_study)
model = Sequential()
model.add(Dense(input_dim=2, output_dim=10))
model.add(Activation("tanh"))
model.add(Dense(output_dim=1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy",
              optimizer="sgd", metrics=["accuracy"])
model.fit(X, Y, nb_epoch=30, batch_size=32)
# 結果を表示
output_X_test = [
    (1, 1),
    (0, 0),
    (1, 0),
    (0, 1),
]
results = model.predict_proba(np.array(output_X_test))

for X, result in zip(output_X_test, results):
    print('input', X, 'prediction', result[0], 'answer', answer_dict[X])
