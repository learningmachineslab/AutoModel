from sklearn.model_selection import RandomizedSearchCV


class AutoModel:
    def __init__(self, model):
        """
        Parameters:
        -----------
        model: a sklern-like model, either a wrapped class or one of
            sklearns classifiers
        """
        self.model = model

    def search_fit(
            self,
            X,
            y,
            param_dict,
            n_iter=10,
            n_jobs=1,
            cv=None,
            random_state=None,
            score_func=None,
    ):
        """
        Parameters:
        -----------
            X: the dataset to fit the model to, should be a partitioned
                "training-set" of the available data
            y: class labels for each data-point
            param_dict: dictionary of parameters for the given model
            n_iter: number of randomized runs
            n_jobs: number of parallel jobs to run the search on
            cv: int, cross-validation generator or an iterable; number of
                folds or object to generate those folds. See RandomizedSearchCV
                docs for a more in-depth description
            random_state: state int to ensure reproducability
        """
        try:
            model_score = getattr(self.model, "score", None)
            if not callable(model_score):
                has_score = False
            else:
                has_score = True
        except AttributeError:
            has_score = False

        if not has_score and score_func is None:
            raise NotImplementedError(
                "If no score_func is supplied, model must have .score method")
        elif not has_score:
            scoring = score_func
        else:
            scoring = None

        rsCV = RandomizedSearchCV(
            self.model,
            param_dict,
            n_iter=n_iter,
            n_jobs=n_jobs,
            cv=cv,
            random_state=random_state,
            scoring=scoring
        )
        rsCV.fit(X, y)
        return rsCV


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets.samples_generator import make_blobs
    import numpy as np

    X, y = make_blobs()
    param_dict = {"C": np.linspace(1e-3, 3, 4)}
    am = AutoModel(LogisticRegression())
    rs = am.search_fit(X, y, param_dict)
    print(rs.best_score_)
