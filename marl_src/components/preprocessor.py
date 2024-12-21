from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Preprocessor:
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.encoder = OneHotEncoder(categories=[list(range(num_classes))])

    def transform(self, data):
        data = np.array(data).reshape(-1, 1)
        return self.encoder.fit_transform(data).toarray()

    def inverse(self, data):
        return self.encoder.inverse_transform(data).flatten()
