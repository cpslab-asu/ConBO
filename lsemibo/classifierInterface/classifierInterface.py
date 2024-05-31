from abc import ABC, abstractmethod


class ClassifierStructure(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def fit_classifier(self, x_train, y_train):
        """Method to fit gpr Model

        Args:
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.

        
        """
        raise NotImplementedError

    @abstractmethod
    def predict_classifier(self, x_test):
        """Method to predict mean and std_dev from gpr model

        Args:
            x_train: Samples from Training set.
            

        Returns:
            mean
            std_dev
        """
        raise NotImplementedError
