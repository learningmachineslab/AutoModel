
class Wrapper():
    """
    The base class for all wrapper classes.

    All functions should return NotImplementedErrors.

    This class is used for importing data into the
    class structure.

    To initialize the model, for example, one would call the wrapper
    """
    
    def __init__(self, model, model_params):
        self.model = model
        self.model_params = model_params

    def fit(self):
        raise NotImplementedError('This function has not been implemented yet.')
    
    def score(self):
        raise NotImplementedError('This function has not been implemented yet.')

    def predict(self):
        raise NotImplementedError('This function has not been implemented yet.')

    def predict_proba(self):
        raise NotImplementedError('This function has not been implemented yet.')

    def decision_function(self):
        raise NotImplementedError('This function has not been implemented yet.')

    def inverse_transform(self):
        raise NotImplementedError('This function has not been implemented yet.')
