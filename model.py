import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

JOBS = 4

class Model:
    def __init__(self):
        self._scaler = StandardScaler()
        self._model = None

    @staticmethod
    def load(name):
        m = Model()
        with open(name, 'rb') as f:
            x = pickle.load(f)
            m._scaler = x['scale']
            m._model = x['model']
        return m

    def store(self, name):
        with open(name, 'wb') as f:
            pickle.dump({'scale': self._scaler,
                         'model': self._model}, f)

    def train(self, C, X, y):
        self._model = GridSearchCV(LinearSVC(), {'C': C}, n_jobs=JOBS)
        self._model.fit(self._scaler.fit_transform(X), y)

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))

    
if __name__ == '__main__':
    import tempfile
    import os
    
    m = Model()
    m.train([1, 10],
            [[1],[3],[10],[1],[2],[3],[4],[5],[5]],
            [1, 0, 1, 1, 1, 1, 0, 0, 1])
    assert(m.predict([[1]]) == [1])
    with tempfile.NamedTemporaryFile(mode='w+b', delete=True) as f:
        m.store(f.name)
        n = Model.load(f.name)
        assert(n.predict([[1]]) == [1])
