import os
import cv2
import imghdr
import keras
from keras.optimizers import Adam;
import keras.metrics as metrics 
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

image_exts = ["jpeg", "jpg"] # bmp and png can run
raw_data_path = "./raw_data"
train_dir = './train'
test_dir = './test'
image_size = 128
num_classes = 2 # classifier only 1 and 2

# remove image that exts is not in image_exts
# Run once we get a new data sets
# for f in os.listdir(raw_data_path):
#     for img_path in os.listdir(os.path.join(raw_data_path, f)):
#         image_path = os.path.join(raw_data_path, f, img_path)
#         try:
#             image = cv2.imread(image_path)
#             exts = imghdr.what(image_path)
#             if(exts not in image_exts):
#                 os.remove(image_path)
#         except:
#             os.remove(image_path)

# label the data
def data_to_array(path):
    label = []
    images = []
    
    for folder in os.listdir(path):
        for image in os.listdir(os.path.join(path, folder)):
            if(folder == "1"):
                label.append([1, 0])
            elif(folder == "2"):
                label.append([0, 1])
            img = cv2.imread(os.path.join(path, folder, image), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            # run only once when new image add
            # if(len(img.shape)==2): 
            #     os.remove(os.path.join(path, folder, image))
            #     continue; 
            images.append(img)

    return images, label

x_train, y_train = data_to_array(train_dir)
x_test, y_test = data_to_array(test_dir)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train.astype('float32')
x_test.astype('float32')
x_train = np.divide(x_train, 255, out=x_train, casting='unsafe')
x_test = np.divide(x_test, 255, out=x_test, casting='unsafe')
# dataset = tf.keras.utils.image_dataset_from_directory(raw_data_path)
# dataset = dataset.map(lambda x, y : (x/255, y))
# scaled_iterator = dataset.as_numpy_iterator()
# batch = scaled_iterator.next()

# train_size = int(len(dataset)*0.7)
# val_size = int(len(dataset)*0.2)
# test_size = int(len(dataset)*0.1)

# train_data = dataset.take(train_size)
# val_data = dataset.skip(train_size).take(val_size)
# test_data= dataset.skip(train_size+val_size).take(test_size)

model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),
    Dense(16),
    Flatten(),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=["acc"])

batch_size = 32
epochs = 50

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
#print(history.history)
plt.figure(figsize=(10,4))
plt.subplot(121),
plt.title('model accuracy');plt.ylabel('accuracy');plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.plot(history.history['acc']);plt.plot(history.history['val_acc'])

plt.subplot(122)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.plot(history.history['loss']);plt.plot(history.history['val_loss'])
plt.show()

# Testing 
test_path = "./test"
for folder in os.listdir(test_path):
    for image in os.listdir(os.path.join(test_path, folder)):
        file = os.path.join(test_path, folder, image)
        if imghdr.what(file) in image_exts:
            img = cv2.imread(file, cv2.COLOR_BGR2RGB)
            ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            rimg = np.array(img)
            rimg.astype('float32')
            rimg = np.divide(rimg, 255, out=rimg, casting="unsafe")
            rimg = np.reshape(rimg, (1, 128, 128, 3))
            predict = model.predict(rimg)
            label = ['1', '2']
            result = label[np.argmax(predict)]
            print(predict)
            print('real : '+str(folder))
            print('predict : '+str(result))
            plt.imshow(ori)
            plt.text(0, 0, 'real : '+str(folder)+' predict : '+str(result))
            plt.show()