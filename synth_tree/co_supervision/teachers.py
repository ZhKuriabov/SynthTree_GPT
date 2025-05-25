from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np

class TeacherModelTrainer:
    def __init__(self, task='classification', model_names=None):
        self.task = task
        self.models = []
        self.model_names = model_names or ['rf', 'gb', 'mlp', 'lr']

    def train_models(self, X, y):
        model_defs = {
            'rf': RandomForestClassifier() if self.task == 'classification' else RandomForestRegressor(),
            'gb': GradientBoostingClassifier() if self.task == 'classification' else GradientBoostingRegressor(),
            'mlp': MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(15, 2), random_state=1)
                   if self.task == 'classification' else MLPRegressor(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(15, 2), random_state=1),
            'lr': LogisticRegression() if self.task == 'classification' else LinearRegression()
        }

        self.models = [model_defs[name] for name in self.model_names if name in model_defs]
        for model in self.models:
            model.fit(X, y)

    def sort_models_by_score(self, X, y):
        if self.task == 'classification':
            scored = [(model, roc_auc_score(y, model.predict(X))) for model in self.models]
        else:
            scored = [(model, r2_score(y, model.predict(X))) for model in self.models]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def predict_all(self, X):
        return np.column_stack([m.predict(X) for m in self.models])
