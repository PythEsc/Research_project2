from abc import ABC, abstractmethod

from importer.database.database_access import DataStorage


class NeuralNetwork(ABC):
    @staticmethod
    @abstractmethod
    def predict(content: list) -> list:
        """
        This method predicts the Facebook reactions for a single post

        :param content: A list containing the contents of multiple Facebook post
        :return: A list of arrays containing the ratio of reactions ['LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY'] per post
        """
        pass

    @staticmethod
    @abstractmethod
    def train(db: DataStorage, sample_percentage: float = 0.2, required_mse: float = 0.3, restore: bool = False):
        """
        This method will start a new model and train it until either the required_accuracy is reached or the user stops
        training

        :param db: The DataStorage that will be used to get training data
        :param sample_percentage: Percentage of how much data should be used for evaluating
        :param required_mse: The required maximum mean squared error at which the training will stop
        :param restore: Boolean to restore latest checkpoint or restart training.
        """
        pass
