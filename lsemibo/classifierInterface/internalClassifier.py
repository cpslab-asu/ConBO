from .classifierInterface import ClassifierStructure
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

class InternalClassifier(ClassifierStructure):
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

    
        