# -*- coding: utf-8 -*-
"""
Created on Tue Nov 2 23:31:08 2018

@author: rito
"""

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_X = train.iloc[:,1:].values
train_y = train['label'].values
test = test.values
encoder = LabelEncoder()
encoded_y = encoder.fit_transform(train_y)
dummy_y = np_utils.to_categorical(encoded_y)


model = Sequential()

model.add(Dense(15, activation='relu', input_shape=(784,)))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_X, dummy_y, validation_split=0.3, epochs=20, callbacks=[early_stopping_monitor])
pred = model.predict_classes(test)

    

pred_df = pd.DataFrame(columns=['Imageid'],
                            data=np.arange(1,28001))
pred_df.loc[:,'Label'] = pd.Series(pred)

pred_df.head()

pred_df.to_csv('predicted_digits.csv', index=False)
