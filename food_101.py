from __future__ import print_function
from __future__ import division

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense
from keras.layers import Input
import numpy as np

def setup_generator(train_path, test_path, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=dimentions[:2],
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        test_path, # this is the target directory
        target_size=dimentions[:2],
        batch_size=batch_size)

    return train_generator, validation_generator

def load_image(img_path, rescale=1. / 255):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x *= rescale # rescale the same as when trained

    return x


def get_classes(file_path):
    with open(file_path) as f:
        classes = f.read().splitlines()

    return classes

def create_model(dimentions, num_classes):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(
            shape=dimentions))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model_final = Model(input=base_model.input, output=predictions)

    return model_final

def train_model(model_final, train_generator, validation_generator, callbacks):
    model_final.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model_final.fit_generator(train_generator, validation_data=validation_generator,
                              epochs=25, callbacks=[checkpointer],
                              steps_per_epoch=train_generator.samples, validation_steps=validation_generator.samples)

def load_model(model_final, weights_path):
   model_final = create_model()
   model_final.load_weights(weights_path)

   return model_final

if __name__ == '__main__':
    batch_size = 32
    dimentions = (224, 224, 3)

    ### use accordingly
    X_train, X_test = setup_generator('train', 'test', batch_size)

    print(X_train)

    # call backs have to be array
    callbacks = []
    # add a callback
    callbacks.append(ModelCheckpoint(filepath='saved_models/food-101-epoch-{epoch:02d}.hdf5',
                                   verbose=1, save_best_only=True))

    model_final = create_model(dimentions, X_train.num_class)

    ### use accordingly
    #train_model(model_final, X_train, X_test, callbacks)

    #trained_model = load_model(model_final, 'saved_models/food-101-epoch-01.hdf5')
    image = load_image('path_to_image')
    preds = model.predict(image)
    classes = get_classes('meta/classes.txt')
    print("the image is: ", classes([np.argmax(preds)))
