import numpy as np
import logging
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


class Visualize:
    def __init__(self):
        pass

    def plot_loss(self, model, epochs):
        """
        Plot loss values regargin model and its epochs

        :param model: the deep learning compiled model
        :param epochs: the number of epochs to display
        """
        logging.info(">> Plotting loss graph...")
        loss = model.history['loss']
        val_loss = model.history['val_loss']

        epochs = range(epochs)

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()

    def plot_accuracy(self, history):
        """
        Plot the accuracy of a trained model

        :param history: the deep learning trained model
        """
        logging.info(">> Plotting accuracy graph...")
        fig, ax = plt.subplots(2, 1, figsize=(18, 10))
        ax[0].plot(history.history['loss'], color='b', label="Training loss")
        ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
        ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
        legend = ax[1].legend(loc='best', shadow=True)

    def confusion_matrix(self, model, validation_generator, num_of_test_samples, batch_size, classes):
        """
        Plot the confusion matrix for a validation set of data

        :param model: the deep learning compiled model
        :param validation_generator: the InputDataGenerator for validation dataset
        :param num_of_test_samples: the number of test images
        :param batch_size: the batch size of the training
        :param classes: the list of classes names

        Source:
            - https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045
        """
        Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size + 1)
        y_pred = np.argmax(Y_pred, axis=1)

        logging.info(">> Plotting confusion matrix...")
        logging.info(confusion_matrix(validation_generator.classes, y_pred))
        logging.info(">> Classification report...")
        logging.info(classification_report(validation_generator.classes, y_pred, target_names=classes))

