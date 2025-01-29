from abc import ABC, abstractmethod
from numpy.typing import NDArray
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class ClassifierSkeleton(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def fit_classifier(self, x_train:NDArray, y_train:NDArray):
        """Method to fit gpr Model

        Args:
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.

        
        """
        raise NotImplementedError

    @abstractmethod
    def predict_classifier(self, x_test:NDArray):
        """Method to predict mean and std_dev from gpr model

        Args:
            x_train: Samples from Training set.
            

        Returns:
            mean
            std_dev
        """
        raise NotImplementedError


class Classifier:
    def __init__(self, classifier_model:ClassifierSkeleton) -> None:
        self.classifier_model = classifier_model

    def fit(self, x_train:NDArray, y_train:NDArray):
        """ Wrapper to fit user defined gpr model

        Args:
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train
        """
        if len(x_train.shape) != 2:
            raise TypeError(f"Received samples set input: Expected (n, dim) array, received {x_train.shape} instead.")
        if len(y_train.shape) != 1:
            raise TypeError(f"Received evaluations set input: Expected (n,) array, received {y_train.shape} instead.")
        if x_train.shape[0] != y_train.shape[0]:
            raise TypeError(f"x_train, y_train set mismatch. x_train has shape {x_train.shape} and y_train has shape {y_train.shape}")

        self.classifier_model.fit_classifier(x_train, y_train)

    def predict(self, X:NDArray):
        """Wrapper to predict from user defined gpr model

        Args:
            X: Samples for predicting

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array

        Returns:
            mean
            std
        """
        if len(X.shape) != 2:
            raise TypeError(f"Received samples set input: Expected (n, dim) array, received {X.shape} instead.")

        pred = self.classifier_model.predict_classifier(X)

        # assert len(mean.shape) == 1, f"Mean from GPR should be of shape (n, ). Received {mean.shape} instead."
        # assert len(std.shape) == 1, f"std_dev from GPR should be of shape (n, ). Received {std.shape} instead."
        # assert mean.shape == std.shape, f"Mean and std_dev mismatch. Mean has a shape of {mean.shape} and std_dev has a shape of {std.shape}."

        return pred
    

class InternalClassifier(ClassifierSkeleton):
    def __init__(self):
        self.model = SVC(C = 1, kernel = "linear", degree = 6)
        self.xscale = StandardScaler()
        # self.yscale = StandardScaler()

    def fit_classifier(self, x_train, y_train):
        X_scaled = self.xscale.fit_transform(x_train)
        # Y_scaled = self.yscale.fit_transform(y_train)
        self.model.fit(X_scaled, y_train)

    def predict_classifier(self, x_test):
        x_scaled = self.xscale.transform(x_test)
        y_test = self.model.predict(x_scaled)        
        return y_test

    
        