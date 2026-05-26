from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

def train_knn_model(X_train,y_train):

    model = KNeighborsClassifier()
    param_grid = {

        'n_neighbors': [
            3,
            5,
            7,
            9
        ],

        'metric': [
            'euclidean',
            'manhattan',
            'minkowski'
        ],

        'weights': [
            'uniform',
            'distance'
        ]
    }


    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )


    grid_search.fit(X_train, y_train)
    best_model = (grid_search.best_estimator_)


    print('\nBest Parameters:\n')
    print(grid_search.best_params_)
    print('\nBest Accuracy:\n')
    print(grid_search.best_score_)
    model_path = (
        BASE_DIR /
        'models' /
        'knn_model.pkl'
    )
    joblib.dump(best_model,model_path)
    return best_model