import numpy as np
from keras.models import Model
from scipy.spatial import distance
from scipy import linalg

class MahalanobisOutlierDetector:
    """
    An outlier detector which uses an input trained model as feature extractor and
    calculates the Mahalanobis distance as an outlier score.
    """
    def __init__(self, features_extractor: Model):
        self.features_extractor = features_extractor
        self.features = None
        self.features_mean = None
        self.features_covmat = None
        self.features_covmat_inv = None
        self.threshold = None
        
    def _extract_features(self, dataset, steps, verbose) -> np.ndarray:
        """
        Extract features from the base model.
        """

        # If x is a tf.data dataset and steps is None, predict() will run until the input dataset is exhausted.
        # but we still need steps here because it's a distributed dataset
        _, _, embedding = self.features_extractor.predict(dataset, steps=steps, workers=8, verbose=verbose)
        
        return embedding
        
    def _init_calculations(self):
        """
        Calculate the prerequired matrices for Mahalanobis distance calculation.
        """
        self.features_mean = np.mean(self.features, axis=0)
        self.features_covmat = np.cov(self.features, rowvar=False)
        self.features_covmat_inv = linalg.inv(self.features_covmat)
        
    def _calculate_distance(self, x) -> float:
        """
        Calculate Mahalanobis distance for an input instance.
        """
        return distance.mahalanobis(x, self.features_mean, self.features_covmat_inv)
    
    def _infer_threshold(self, verbose):
        """
        Infer threshold based on the extracted features from the training set.
        """
        scores = np.asarray([self._calculate_distance(feature) for feature in self.features])
        mean = np.mean(scores)
        std = np.std(scores)
        self.threshold = mean + 2 * std
        if verbose > 0:
            print("OD score mean:", mean)
            print("OD score std :", std)
            print("OD threshold :", self.threshold)  
            
    def fit(self, dataset, steps, verbose=1):
        """
        Fit detector model.
        """
        self.features = self._extract_features(dataset, steps, verbose)
        self._init_calculations()
        self._infer_threshold(verbose)
        
    def predict(self, dataset, steps, verbose=1) -> np.ndarray:
        """
        Calculate outlier score (Mahalanobis distance).
        """
        features  =  self._extract_features(dataset, steps, verbose)
        scores = np.asarray([self._calculate_distance(feature) for feature in features])
        if verbose > 0:
            print("OD score mean:", np.mean(scores))
            print("OD score std :", np.std(scores))
            print(f"Outliers     :{len(np.where(scores > self.threshold )[0])/len(scores): 1.2%}")

        # get all the labels from the dataset
        
        # doesn't work because it's a distributed DS
        # labels = dataset.map(lambda x: x[1])
        labels = []
        iterator = iter(dataset)
        for (features, label) in iterator:
            labels.append(label)

        labels = np.array(labels)
        print(labels)
        # so all anomalies should have a score higher than 1
        pred = scores > self.threshold

        TP = np.count_nonzero(pred * labels)
        TN = np.count_nonzero((pred - 1) * (labels - 1))
        FP = np.count_nonzero(pred * (labels - 1))
        FN = np.count_nonzero((pred - 1) * labels)

        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print("++++++++++++++++PRECISION++++++++++++")
        print(precision)
        print("++++++++++++++++RECALL++++++++++++")
        print(recall)

            
        # if verbose > 1:
        #     plt.hist(scores, bins=100);
        #     plt.axvline(self.threshold, c='k', ls='--', label='threshold')
        #     plt.xlabel("Mahalanobis distance"); plt.ylabel("Distribution");
        #     plt.show()
            
        return scores