from __future__ import print_function
from __future__ import division

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import argparse


def setup_generator(train_path, test_path, batch_size, dimentions):
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
        target_size=dimentions,
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        test_path, # this is the target directory
        target_size=dimentions,
        batch_size=batch_size)

    return train_generator, validation_generator


def load_image(img_path, dimentions, rescale=1. / 255):
    img = image.load_img(img_path, target_size=dimentions)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x *= rescale # rescale the same as when trained

    return x


def get_classes(file_path):
    with open(file_path) as f:
        classes = f.read().splitlines()

    return classes


def create_model(num_classes, dropout, shape):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(
            shape=shape))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model_final = Model(input=base_model.input, output=predictions)

    return model_final


def train_model(model_final, train_generator, validation_generator, callbacks, args):
    model_final.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model_final.fit_generator(train_generator, validation_data=validation_generator,
                              epochs=args.epoch, callbacks=[callbacks],
                              steps_per_epoch=train_generator.samples,
                              validation_steps=validation_generator.samples)


def load_model(model_final, weights_path, shape):
   model_final = create_model(101, 0, shape)
   model_final.load_weights(weights_path)

   return model_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Food 101 Program')
    parser.add_argument('-m', help='train or inference model', dest='mode',
                        type=str, default='train')
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=32)
    parser.add_argument('-p', help='path of the saved model', dest='model_path',
                        type=str, default='saved_models/food-101-epoch-01.hdf5')
    parser.add_argument('-i', help='path to test image', dest='image_path',
                        type=str, default='')
    parser.add_argument('-e', help='epochs to train the model', dest='epochs',
                        type=int, default=25)
    parser.add_argument('-d', help='decimal value for dropout', dest='dropout',
                        type=float, default=0.0)

    args = parser.parse_args()

    shape = (224, 224, 3)

    if args.mode == 'train':
        X_train, X_test = setup_generator('train', 'test', args.batch_size, shape[:2])

        # debug purposes
        print(X_train)

        # call backs have to be array
        callbacks = []
        # add a callback
        callbacks.append(ModelCheckpoint(filepath='saved_models/food-101-epoch-{epoch:02d}.hdf5',
                                       verbose=1, save_best_only=True))

        model_final = create_model(X_train.num_class, args.dropout, shape)

        train_model(model_final, X_train, X_test, callbacks, args)
    else:
        trained_model = load_model(model_final, args.model_path, shape)
        image = load_image(args.image_path, shape[:2])
        preds = trained_model.predict(image)
        classes = get_classes('meta/classes.txt')
        print("the image is: ", classes([np.argmax(preds)]))
