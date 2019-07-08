
from wrapper import Wrapper

class statsmodelsWrapper(Wrapper):

    def __init__(self, training_data):
        super().__init__(training_data)

    def fit(self):
        pass
    
    def score(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def decision_function(self):
        pass

    def inverse_transform(self):
        pass

    

if __name__ == '__main__':
    print('goodbye cruel world!')
