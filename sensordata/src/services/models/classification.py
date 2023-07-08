import pickle
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from solution.utils.Common import getByKey

""" Set the seed for reproductibility 
    
    Another way to do it, this will set the numpy random seed as well
    RANDOM_SEED = 0
    SEED = np.random.seed(RANDOM_SEED)
"""
SEED = 42


"""
    Models can be added here for validation and testing.
    KEY : can be unique name working as an identifier
    VALUE: Will contain a dictionary with 2 fixed names
        model: This contains the Classifier class
        params: This contains the parameters for hyperparameter tuning using cross validation 
"""
_modelDict = {
    "LR": {
        "model": lm.LogisticRegression(random_state=SEED),
        "params": {
            "penalty": [None, "l2"],
        },
    },
    "GaussianProcess": {
        "model": GaussianProcessClassifier(random_state=SEED),
        "params": {
            "kernel": [1.0 * RBF(1.0)],
            "optimizer": ["fmin_l_bfgs_b", None],
        },
    },
}


class Model:
    def __call__(self, _modelName: str) -> None:
        self.modelName = _modelName
        modelDict = getByKey(_modelDict, self.modelName)
        model = modelDict["model"]

        # Using a pipeline for model with preprocessing models - this is a secure way to do it
        pipelineSteps = [
            ("transform", StandardScaler()),
            ("model", model),
        ]
        self.model = Pipeline(pipelineSteps)
        self.valParams = modelDict["params"]

    def fit(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if params:
            self.model.set_params(**params)

        self.result = self.model.fit(features, target)

    def validate(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray],
        scoring: str = "accuracy",
    ) -> Tuple[Dict[str, Any], float, Dict[str, np.ndarray]]:
        if self.valParams is not None:
            # add prefix to the params to get identified in the pipeline
            valParams = {f"model__{k}": v for k, v in self.valParams.items()}

            # gridsearch for hyperparameter tuning using StratifiedKFold
            m = GridSearchCV(
                estimator=self.model,
                param_grid=valParams,
                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED),
                scoring=scoring,
                n_jobs=-2,
                verbose=0,
            )
            m.fit(X=features, y=target)

            cvResults = {
                k: v
                for k, v in m.cv_results_.items()
                if k in ("params", "mean_test_score", "std_test_score")
            }

            return (
                m.best_params_,
                m.best_score_,
                cvResults,
            )
        else:
            raise Exception(
                "Parameters for this model is 'None', validation can not be performed"
            )

    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.result.predict(features)

    def predictProbability(self, features: pd.DataFrame) -> pd.DataFrame:
        predProb = self.result.predict_proba(features)

        prob = np.max(predProb, axis=1)
        prob = list(map(lambda x: round(x, 5), prob))
        pred = self.result.classes_[np.argmax(predProb, axis=1)]

        return pd.DataFrame(
            {"prediction": pred, "probability": prob}, index=features.index
        )

    def parameters(self) -> Dict[str, Any]:
        return self.model.get_params()

    def name(self) -> str:
        return self.modelName

    def getModel(self) -> Pipeline:
        return self.model

    @staticmethod
    def toPickle(data, file: str):
        """
        This method return the pickled object of this class to be stored in File system/s3/Azure Blob etc.
        """
        pickle.dump(data, open(file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def fromPickle(file: str):
        """
        This method loads the pickled object from a file
        """
        return pickle.load(open(file, "rb"))
