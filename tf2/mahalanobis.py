import numpy as np
from keras.models import Model
from scipy.spatial import distance
from scipy import linalg
import tensorflow as tf

class MahalanobisOutlierDetector:
    """
    An outlier detector which uses an input trained model as feature extractor and
    calculates the Mahalanobis distance as an outlier score.
    """
    def __init__(self, features_extractor: Model):
        self.model = features_extractor
        self.features = None
        self.features_mean = None
        self.features_covmat = None
        self.features_covmat_inv = None
        self.threshold = None
        
    def _extract_features(self, dataset, steps, strategy, verbose) -> np.ndarray:
        """
        Extract features from the base model.
        """

        # If x is a tf.data dataset and steps is None, predict() will run until the input dataset is exhausted.
        # but we still need steps here because it's a distributed dataset
        # _, _, embedding = self.model.predict(dataset, steps=steps, workers=8, verbose=verbose)

        embeddings = []
        @tf.function
        def single_step(images):
            _, _, embedding = self.model(images, training=False)
            return embedding

        iterator = iter(dataset)
        for _ in range(steps):
            images, _ = next(iterator)
            batch_embedding = strategy.run(single_step, (images,))
            batch_embedding = strategy.gather(batch_embedding, axis=0)
            embeddings += list(batch_embedding.numpy())
        
        return np.array(embeddings)
        
    def _init_calculations(self):
        """
        Calculate the prerequired matrices for Mahalanobis distance calculation.
        """
        self.features_mean = np.mean(self.features, axis=0)
        self.features_covmat = np.cov(self.features, rowvar=False)
        self.features_covmat_inv = linalg.inv(self.features_covmat)
        print(self.features.shape)
        print(self.features_mean.shape)
        print(self.features_covmat)
        
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
            
    def fit(self, dataset, steps, strategy, verbose=1):
        """
        Fit detector model.
        """
        self.features = self._extract_features(dataset, steps, strategy, verbose)
        self._init_calculations()
        self._infer_threshold(verbose)
        
    def predict(self, dataset, steps, strategy, verbose=1) -> np.ndarray:
        """
        Calculate outlier score (Mahalanobis distance).
        """
        features  =  self._extract_features(dataset, steps, strategy, verbose)
        scores = np.asarray([self._calculate_distance(feature) for feature in features])
        if verbose > 0:
            print("OD score mean:", np.mean(scores))
            print("OD score std :", np.std(scores))
            print(f"Outliers     :{len(np.where(scores > self.threshold )[0])/len(scores): 1.2%}")

        # get all the labels from the dataset

        @tf.function
        def get_labels(inputs):
            images, labels = inputs
            return tf.math.argmax(labels, axis=1)
        
        labels = []
        iterator = iter(dataset)
        for _ in range(steps):
            batch_label = strategy.run(get_labels, args=(next(iterator),))
            batch_label = strategy.gather(batch_label, axis=0)
            labels += list(batch_label.numpy())

        labels = np.array(labels)
        # print(labels)
        # print(labels.shape)
        # so all anomalies should be 1
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