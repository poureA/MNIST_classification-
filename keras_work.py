from keras.datasets import mnist
(xtr,ytr),(xts,yts) = mnist.load_data()
print(xtr.shape,ytr.shape,xts.shape,yts.shape)
xtr = xtr.reshape((60000,28*28))
xtr = xtr.astype('float32')/255
xts = xts.reshape((10000,28*28))
xts = xts.astype('float32')/255
from keras.utils import to_categorical
ytr = to_categorical(ytr)
yts = to_categorical(yts)
from keras import models
from keras import layers
alg = models.Sequential()
alg.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
alg.add(layers.Dense(10,activation='softmax'))
alg.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
alg.fit(xtr,ytr,epochs=5,batch_size=128)
test_loss,test_score = alg.evaluate(xts,yts)
print(test_score)
exit = input('enter any key to exit :')