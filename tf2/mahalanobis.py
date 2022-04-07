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
    def __init__(self, features_extractor: Model, store_pickle = False):
        self.model = features_extractor
        self.features = None
        self.classes = None
        self.class_names = None
        # self.features_mean = None
        # self.features_covmat = None
        # self.features_covmat_inv = None
        self.class_params = None
        # self.threshold = None,
        # self.fit_scores = None
        self.class_scores = None
        self.closs_thresholds = None
        self.pred_scores = None
        self.store_pickle = store_pickle 

    def _extract_features(self, dataset, steps, strategy, verbose) -> np.ndarray:
        """
        Extract features from the base model.
        """

        # If x is a tf.data dataset and steps is None, predict() will run until the input dataset is exhausted.
        # but we still need steps here because it's a distributed dataset
        # _, _, embedding = self.model.predict(dataset, steps=steps, workers=8, verbose=verbose)

        @tf.function
        def single_step(inputs):
            images, lbls, classes = inputs
            _, _, embedding = self.model(images, training=False)
            return embedding, tf.math.argmax(lbls, axis=1), classes

        labels = []
        features = []
        classes = []
        #
        iterator = iter(dataset)
        for s in range(steps):
            batch_embedding, batch_label, batch_classes = strategy.run(single_step, args=(next(iterator),))
            if tf.version.VERSION == '2.7.0':
                batch_embedding = strategy.gather(batch_embedding, axis=0)
                batch_label = strategy.gather(batch_label, axis=0)
                batch_classes = strategy.gather(batch_classes, axis=0)
            labels += list(batch_label.numpy())
            features += list(batch_embedding.numpy())
            classes += list(batch_classes.numpy())
            logging.info("Completed feature extraction for step {}/{}".format(s+1, steps))

        return np.array(features), np.array(labels), np.array(classes)
        
    def _init_calculations(self):
        """
        Calculate the prerequired matrices for Mahalanobis distance calculation.
        """
        # we calculate the mean and covmant for every class separately
        self.class_params = {}

        # self.n_classes = len(set(self.classes)) 
        for c in self.class_names:
            mask = self.classes == c
            # only get the features for the relevant class
            class_features = self.features[mask]
            
            features_mean = np.mean(class_features, axis=0)
            features_covmat = np.cov(class_features, rowvar=False)
            features_covmat_inv = linalg.inv(features_covmat)
            logging.info("params for class {}".format(c))
            logging.info("features shape: {}".format(class_features.shape))
            logging.info("features mean shape: {}".format(features_mean.shape))
            logging.info("features covmat")
            logging.info(features_covmat)
            self.class_params[c] = {
                'features_mean': features_mean,
                'features_covmat': features_covmat,
                'features_covmat_inv': features_covmat_inv
            }
        
    def _calculate_distance(self, x, cls) -> float:
        """
        Calculate Mahalanobis distance for an input instance.
        """
        return distance.mahalanobis(x, self.class_params[cls]['features_mean'], self.class_params[cls]['features_covmat_inv'])
    
    def _infer_threshold(self, verbose):
        """
        Infer threshold based on the extracted features from the training set.
        """
        self.class_scores = {}
        for i, feature in enumerate(self.features):
            cls = self.classes[i]
            fit_score = self._calculate_distance(feature, cls)

            if cls in self.class_scores:
                self.class_scores[cls].append(fit_score)
            else:
                self.class_scores[cls] = [fit_score]


        self.class_thresholds = {}

        for cls,scores in self.class_scores.items():
            mean = np.mean(scores)
            std = np.std(scores)
            self.class_thresholds[cls] ={
                'mean': mean,
                'std': std,
                'threshold': mean + 2.0 * std
            }

        
        # if verbose > 0:
        #     logging.info("OD score in infer mean {}".format(np.mean(fit_scores)))
        #     logging.info("OD score in infer std {}".format(np.std(fit_scores)))
        #     logging.info("OD threshold {}".format(threshold))

    def fit(self, dataset, steps, strategy, verbose=1):
        """
        Fit detector model.
        """
        logging.info("Inferring threshold for OD score...")
        self.features, labels, self.classes = self._extract_features(dataset, steps, strategy, verbose)
        self.class_names = list(set(self.classes))
        self._init_calculations()
        self._infer_threshold(verbose)
        #

        if self.store_pickle:
            scores_labels = dict(zip(self.class_scores, labels))
            with open('/home/q373612/LMU/simclr/tf2/last_mahalanobis_fit.pickle', 'wb') as handle:
                pickle.dump((scores_labels, self.threshold), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, dataset, steps, strategy, verbose=1) -> np.ndarray:
        """
        Calculate outlier score (Mahalanobis distance).
        """

        logging.info("Extracting features using eval data and using threshold...")
        features, labels, classes = self._extract_features(dataset, steps, strategy, verbose)
        # so all anomalies should be 1
        # assert self.features is not None

        # first we have to find the closest class as judged by mahalanobis

        pred_anom = []
        pred_class = []
        for feature in features:
            pred_scores = []
            for c in self.class_names:
                pred_scores.append(self._calculate_distance(feature, c))
            pred_class_idx = np.argmin(pred_scores)

            # for full compatibility with paper
            # add small noice to sample
            # and recompute score

            score = pred_scores[pred_class_idx]
            c = self.class_names[pred_class_idx]
            pred_class.append(c)
            threshold = self.class_thresholds[c]['threshold']
            pred_anom.append(score > threshold)

        pred_anom = np.array(pred_anom)
        pred_class = np.array(pred_class)


        # pred_scores = np.asarray([self._calculate_distance(feature) for feature in features])
        # if verbose > 0:
        #     logging.info("OD score in predict mean {}".format(np.mean(self.pred_scores)))
        #     logging.info("OD score in predict std {}".format(np.std(self.pred_scores)))
        #     logging.info(f"Outliers     :{len(np.where(self.pred_scores > self.threshold )[0])/len(self.pred_scores): 1.2%}")

        # pred = self.pred_scores > self.threshold
        # #

        # if self.store_pickle:
        #     scores_labels = dict(zip(self.pred_scores, labels))
        #     with open('/home/q373612/LMU/simclr/tf2/last_mahalanobis_pred.pickle', 'wb') as handle:
        #         pickle.dump((scores_labels, self.threshold), handle, protocol=pickle.HIGHEST_PROTOCOL)

        class_acc = pred_class == classes
        class_acc = np.sum(class_acc) / len(pred_class)

        TP = np.count_nonzero(pred_anom * labels)
        TN = np.count_nonzero((pred_anom - 1) * (labels - 1))
        FP = np.count_nonzero(pred_anom * (labels - 1))
        FN = np.count_nonzero((pred_anom - 1) * labels)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        logging.info("++++++++++++++++PRECISION++++++++++++")
        logging.info(precision)
        logging.info("++++++++++++++++RECALL++++++++++++")
        logging.info(recall)
        logging.info("++++++++++++++++CLASS ACCURACY++++++++++++")
        logging.info(class_acc)

            
        # if verbose > 1:
        #     plt.hist(scores, bins=100);
        #     plt.axvline(self.threshold, c='k', ls='--', label='threshold')
        #     plt.xlabel("Mahalanobis distance"); plt.ylabel("Distribution");
        #     plt.show()
            
        return self.pred_scores
