import warnings
warnings.filterwarnings('ignore')
import os
from minio import Minio
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import shutil
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class LoadAndPrepareImageData:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    @staticmethod
    def getImageDataFromMinIO():
        # ENV Variables
        minio_address = str(os.environ['MINIO_ADDRESS'])
        minio_port = str(os.environ['MINIO_PORT'])
        minio_access_key = str(os.environ['MINIO_ACCESS'])
        minio_secret_key = str(os.environ['MINIO_SECRET'])
        bucket_name = str(os.environ['MINIO_BUCKET_NAME'])
        object_name = str(os.environ['MINIO_OBJECT_NAME'])

        minioClient = Minio(
            '{0}:{1}'.format(minio_address, minio_port),
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
        )

        minioClient.fget_object(bucket_name, object_name, object_name)
        with zipfile.ZipFile(object_name, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(object_name)

        train_data_path = "train"
        test_data_path = "test"

        return train_data_path, test_data_path

    def dataAugmentationAndClassLabels(self, train_data_location):
        # Image Augmentation: Generate more tensor images for balanced training images per class
        train = ImageDataGenerator(rescale=1./255,  # Scale image pixel values in between [0, 1]
                                   rotation_range=10,  # Rotate images in 10 degrees range
                                   horizontal_flip=True,  # Random flip image horizontally
                                   shear_range=0.2,  # Distort the image
                                   fill_mode='nearest',  # Fill points outside boundary
                                   zoom_range=0.1)  # 10% zoom
        train_dataset = train.flow_from_directory(train_data_location,
                                                  target_size=(150, 150),
                                                  batch_size=self.batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)
        print("The classification labels are: ")
        print(train_dataset.class_indices)
        return train_dataset


class SequentialKerasModel:
    def __init__(self, optimizer, loss, epochs, train_dataset):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.training_dataset = train_dataset

    def defineModel(self):
        covid_class_model = Sequential()
        covid_class_model.add(Conv2D(128, kernel_size=7, activation='relu', input_shape=(150, 150, 3)))
        covid_class_model.add(MaxPool2D(pool_size=(4, 4), strides=(2, 2)))
        covid_class_model.add(Conv2D(64, kernel_size=5, activation='relu'))
        covid_class_model.add(MaxPool2D(pool_size=(4, 4), strides=(2, 2)))
        covid_class_model.add(Flatten())
        covid_class_model.add(Dense(128, activation='relu'))
        covid_class_model.add(Dense(3, activation='softmax'))
        covid_class_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['acc'])
        print("Summary of the COVID-19 classification model: \n")
        print(covid_class_model.summary())
        return covid_class_model

    @staticmethod
    def defineCallbacks():
        early_stop = EarlyStopping(monitor="val_loss",
                                   min_delta=0,
                                   patience=0,
                                   verbose=0,
                                   mode="auto",
                                   baseline=None,
                                   restore_best_weights=False,)

        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.2,
                                                 patience=5,
                                                 min_lr=0.001)

        model_check_point = ModelCheckpoint('best_model.hdf5',
                                            monitor='val_accuracy',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
        return early_stop, reduce_learning_rate, model_check_point

    def trainModel(self, covid_class_model, early_stop, reduce_learning_rate, model_check_point):
        covid_class_model_history = covid_class_model.fit(self.training_dataset,
                                                          epochs=self.epochs,
                                                          callbacks=[early_stop, reduce_learning_rate,
                                                                     model_check_point])

        return covid_class_model_history


class PlotLossAndAccuracy:
    def __init__(self, seq_model_hist, start_point):
        self.seq_model_hist = seq_model_hist
        self.start_point = start_point

    def saveModelPlotLocally(self):
        train_acc = self.seq_model_hist.history['acc']
        train_loss = self.seq_model_hist.history['loss']
        epoch_count = len(train_acc) + self.start_point
        epochs = []
        for i in range(self.start_point, epoch_count):
            epochs.append(i + 1)
        index_loss = np.argmin(train_loss)
        val_lowest = train_loss[index_loss]
        index_acc = np.argmax(train_acc)
        acc_highest = train_acc[index_acc]
        plt.style.use('fivethirtyeight')
        train_loss_label = 'Epoch with least loss= ' + str(index_loss + 1 + self.start_point)
        train_acc_label = 'Epoch with best accuracy= ' + str(index_acc + 1 + self.start_point)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
        axes[0].plot(epochs, train_loss, 'g', label='Training loss')
        axes[0].scatter(index_loss + 1 + self.start_point, val_lowest, s=150, c='blue', label=train_loss_label)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[1].plot(epochs, train_acc, 'r', label='Training Accuracy')
        axes[1].scatter(index_acc + 1 + self.start_point, acc_highest, s=150, c='blue', label=train_acc_label)
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        plt.savefig('training_loss_accuracy_plots.png')


class EvaluateKerasModel:
    def __init__(self, seq_model, testing_images_location, batch_size):
        self.seq_model = seq_model
        self.testing_images_location = testing_images_location
        self.batch_size = batch_size

    def evaluateModelAgainstTestData(self):
        test_dataset = ImageDataGenerator(rescale=1./255)
        testing_images = test_dataset.flow_from_directory(self.testing_images_location,
                                                          target_size=(150, 150),
                                                          batch_size=self.batch_size,
                                                          shuffle=False)
        testing_steps_per_epoch = np.math.ceil(testing_images.samples / testing_images.batch_size)

        predictions = self.seq_model.predict(testing_images, steps=testing_steps_per_epoch)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = testing_images.classes
        class_labels = list(testing_images.class_indices.keys())

        print("Covid-19 Classification Model Accuracy: ", accuracy_score(true_classes, predicted_classes), '\n')
        print("The confusion matrix is: \n")
        print(confusion_matrix(true_classes, predicted_classes), '\n')
        print("The classification report is: \n")
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    def saveKerasModel(self):
        self.seq_model.save("my_covid_classification_model.h5")


class Execute:
    def __init__(self, batch_size: int, epochs: int, optimizer, loss):
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss

    def startExecution(self):
        # Fetch data and prepare for model training
        train_data_path, test_data_path = LoadAndPrepareImageData.getImageDataFromMinIO()
        prepare_data_obj = LoadAndPrepareImageData(self.batch_size)
        training_images = prepare_data_obj.dataAugmentationAndClassLabels(train_data_path)

        # Model Training
        seq_keras_model_obj = SequentialKerasModel(self.optimizer, self.loss, self.epochs, training_images)
        covid_class_model = seq_keras_model_obj.defineModel()
        early_stop, reduce_learning_rate, model_check_point = seq_keras_model_obj.defineCallbacks()
        model_results = seq_keras_model_obj.trainModel(covid_class_model,
                                                       early_stop,
                                                       reduce_learning_rate,
                                                       model_check_point)

        # Plot Model Results
        plot_obj = PlotLossAndAccuracy(model_results, 0)
        plot_obj.saveModelPlotLocally()

        # Evaluate the Model
        evaluate_obj = EvaluateKerasModel(covid_class_model, test_data_path, self.batch_size)
        evaluate_obj.evaluateModelAgainstTestData()
        evaluate_obj.saveKerasModel()
        time.sleep(3)
        try:
            shutil.rmtree(train_data_path)
            shutil.rmtree(test_data_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == "__main__":
    # Optimizer: adam
    # Loss function: categorical_crossentropy
    # Batch size: 32
    # Epochs: 30
    Execute(32, 30, 'adam', 'categorical_crossentropy').startExecution()
