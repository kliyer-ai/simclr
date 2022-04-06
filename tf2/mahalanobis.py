import numpy as np
from tensorflow.keras.models import Model
from scipy.spatial import distance
from scipy import linalg
import tensorflow as tf
from absl import logging
import pickle

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
        self.threshold = None,
        self.fit_scores = None
        self.pred_scores = None

    def _extract_features(self, dataset, steps, strategy, verbose) -> np.ndarray:
        """
        Extract features from the base model.
        """

        # If x is a tf.data dataset and steps is None, predict() will run until the input dataset is exhausted.
        # but we still need steps here because it's a distributed dataset
        # _, _, embedding = self.model.predict(dataset, steps=steps, workers=8, verbose=verbose)

        @tf.function
        def single_step(inputs):
            images, lbls = inputs
            _, _, embedding = self.model(images, training=False)
            return embedding, tf.math.argmax(lbls, axis=1)

        labels = []
        features = []
        #
        iterator = iter(dataset)
        for s in range(steps):
            batch_embedding, batch_label = strategy.run(single_step, args=(next(iterator),))
            #batch_label = strategy.gather(batch_label, axis=0)
            labels += list(batch_label.numpy())
            features += list(batch_embedding.numpy())
            logging.info("Completed feature extraction for step {}/{}".format(s+1, steps))

        return np.array(features), np.array(labels)
        
    def _init_calculations(self):
        """
        Calculate the prerequired matrices for Mahalanobis distance calculation.
        """
        self.features_mean = np.mean(self.features, axis=0)
        self.features_covmat = np.cov(self.features, rowvar=False)
        self.features_covmat_inv = linalg.inv(self.features_covmat)
        logging.info("features shape: {}".format(self.features.shape))
        logging.info("features mean shape: {}".format(self.features_mean.shape))
        logging.info("features covmat")
        logging.info(self.features_covmat)
        
    def _calculate_distance(self, x) -> float:
        """
        Calculate Mahalanobis distance for an input instance.
        """
        return distance.mahalanobis(x, self.features_mean, self.features_covmat_inv)
    
    def _infer_threshold(self, verbose):
        """
        Infer threshold based on the extracted features from the training set.
        """
        self.fit_scores = np.asarray([self._calculate_distance(feature) for feature in self.features])
        mean = np.mean(self.fit_scores)
        std = np.std(self.fit_scores)
        self.threshold = mean + 2.0 * std
        if verbose > 0:
            logging.info("OD score in infer mean {}".format(np.mean(self.fit_scores)))
            logging.info("OD score in infer std {}".format(np.std(self.fit_scores)))
            logging.info("OD threshold {}".format(self.threshold))

    def fit(self, dataset, steps, strategy, verbose=1):
        """
        Fit detector model.
        """
        logging.info("Inferring threshold for OD score...")
        self.features, labels = self._extract_features(dataset, steps, strategy, verbose)
        self._init_calculations()
        self._infer_threshold(verbose)
        #
        scores_labels = dict(zip(self.fit_scores, labels))
        with open('last_mahalanobis_fit.pickle', 'wb') as handle:
            pickle.dump((scores_labels, self.threshold), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, dataset, steps, strategy, verbose=1) -> np.ndarray:
        """
        Calculate outlier score (Mahalanobis distance).
        """

        logging.info("Extracting features using eval data and using threshold...")
        features, labels = self._extract_features(dataset, steps, strategy, verbose)
        # so all anomalies should be 1
        # assert self.features is not None
        self.pred_scores = np.asarray([self._calculate_distance(feature) for feature in features])
        if verbose > 0:
            logging.info("OD score in predict mean {}".format(np.mean(self.pred_scores)))
            logging.info("OD score in predict std {}".format(np.std(self.pred_scores)))
            logging.info(f"Outliers     :{len(np.where(self.pred_scores > self.threshold )[0])/len(self.pred_scores): 1.2%}")

        pred = self.pred_scores > self.threshold
        #
        scores_labels = dict(zip(self.pred_scores, labels))
        with open('/home/q373612/LMU/simclr/tf2/last_mahalanobis_pred.pickle', 'wb') as handle:
            pickle.dump((scores_labels, self.threshold), handle, protocol=pickle.HIGHEST_PROTOCOL)

        TP = np.count_nonzero(pred * labels)
        TN = np.count_nonzero((pred - 1) * (labels - 1))
        FP = np.count_nonzero(pred * (labels - 1))
        FN = np.count_nonzero((pred - 1) * labels)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        logging.info("++++++++++++++++PRECISION++++++++++++")
        logging.info(precision)
        logging.info("++++++++++++++++RECALL++++++++++++")
        logging.info(recall)

            
        # if verbose > 1:
        #     plt.hist(scores, bins=100);
        #     plt.axvline(self.threshold, c='k', ls='--', label='threshold')
        #     plt.xlabel("Mahalanobis distance"); plt.ylabel("Distribution");
        #     plt.show()
            
        return self.pred_scores