from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.layers import LeakyReLU
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import deepcut as dc
import numpy as np
from gensim.models import Word2Vec


class osh_keras:


    def keras_model(self):

        model = Sequential()
        model.add(Dense(10 ,input_dim= 100, kernel_initializer='normal'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(10, kernel_initializer='normal'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))

        # Compile model
        # for binary classification
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def pre_processing(self , filepath= None):

        self.filepath = filepath


        dataframe = pd.read_csv(self.filepath)
        dataset = dataframe.values

        x = dataset[:, 1: -1].astype(float)
        y = dataset[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        ## Change label to one hot encoding
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_Y_train = encoder.transform(y_train)
        encoded_Y_train = np_utils.to_categorical(encoded_Y_train)

        encoder.fit(y_test)
        encoded_Y_test = encoder.transform(y_test)
        encoded_Y_test = np_utils.to_categorical(encoded_Y_test)


        return X_train, X_test, encoded_Y_train, encoded_Y_test

    def test_model(self):

        text = input('To test model loading, Fill your query: ').lower()

        # if text == '':
        #     print ('Your query is Empty!!!, Please fill it again')
        #     pass
        # else:

        token_text = dc.tokenize(text, custom_dict=['ไม้ฝา', 'เฌอร่า', 'ไม้'])
        query = list(filter((' ').__ne__, token_text))
        word2vec_model = Word2Vec.load("1_word2vec_model/query_window1.model")

        print(query)

        vec = np.zeros((1, 100))

        for i in query:

            if i in word2vec_model.wv.index2word:

                vec += word2vec_model.wv.word_vec(i)
            else:
                pass
                print(i, ' :not found in dic')

        return vec


# define class
test = osh_keras()


model = test.keras_model()
X_train,X_test,encoded_Y_train, encoded_Y_test = test.pre_processing(filepath = "vectorized.csv")


# For data set that large enough use validation

history=model.fit(X_train, encoded_Y_train, batch_size=10, nb_epoch=50, validation_split = 0.2, verbose=1)
#history = model.fit(X_train, encoded_Y_train, batch_size=10, nb_epoch=50)

# return array
loss , acc = model.evaluate(X_test, encoded_Y_test, batch_size=10)


print ('Test loss: ', '{0:.4f}'.format(loss))
print ('Test accuracy: ', '{0:.2f}'.format(acc * 100), '%' )




## To test model!!
vec = test.test_model()
classes = model.predict(vec, batch_size=10)

print(classes)

