import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns

california = fetch_california_housing(as_frame=True)
df = california.frame
df
df.describe()

import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns


# Initialiser l'expérience MLflow
mlflow.set_experiment("mongi_experiment_s3")
with mlflow.start_run(run_name="main_run_project"):
    with mlflow.start_run(run_name="data_exploration", nested=True) as run:
        # Charger le jeu de données
        california = fetch_california_housing(as_frame=True)
        df = california.frame
        mlflow.log_param("dataset_shape", df.shape)

        # Analyse descriptive
        desc = df.describe()
        print(desc)
        mlflow.log_text(desc.to_string(), "description.txt")
        mlflow.log_artifact("california-housing.py")

        # Visualisation des distributions (exemples)
        plt.figure(figsize=(10, 6))
        sns.histplot(df['MedHouseVal'], bins=50)
        plt.title('Distribution des Prix des Maisons')
        plt.savefig("histogram_price.png")
        mlflow.log_artifact("histogram_price.png")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="MedInc", y="MedHouseVal", data=df)
        plt.title("Relation entre Revenu Médian et Prix")
        plt.savefig("scatter_income_price.png")
        mlflow.log_artifact("scatter_income_price.png")

        # Log des infos, observations, etc.
        mlflow.log_text("Observation: Le prix des maisons a une distribution non normale...", "data_insights.txt")
        print("Exploration terminée")

        
    # from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics import accuracy_score, classification_report

    # with mlflow.start_run(run_name="logistic_regression_baseline", nested=True) as run:
    #     # Préparation des données pour la classification (exemple: prix > moyenne devient 1, sinon 0)
    #     df['target_class'] = (df['MedHouseVal'] > df['MedHouseVal'].mean()).astype(int)
    #     X = df.drop(['MedHouseVal','target_class'], axis=1)
    #     y = df['target_class']
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #     # Mise à l'échelle des features
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.transform(X_test)

    #     # Entraînement du modèle de régression logistique
    #     model = LogisticRegression(solver='liblinear', random_state=42)  #solver to prevent warnings
    #     model.fit(X_train_scaled, y_train)

    #     # Prédiction et évaluation
    #     y_pred = model.predict(X_test_scaled)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     report = classification_report(y_test, y_pred, output_dict=True)

    #     # Log des paramètres, métriques et modèle
    #     mlflow.log_param("solver", "liblinear")
    #     mlflow.log_metric("accuracy", accuracy)
    #     mlflow.log_dict(report, "classification_report.json")
    #     mlflow.sklearn.log_model(model, "logistic_model")
    #     print("Modèle Logistic Regression Entraîné")


    # from sklearn.linear_model import LinearRegression, Ridge
    # from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    # from sklearn.metrics import mean_squared_error, r2_score
    # import numpy as np

    # models = {
    #     "LinearRegression": LinearRegression(),
    #     "Ridge": Ridge(alpha=1.0),
    #     "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    #     "RandomForest": RandomForestRegressor(n_estimators=30, random_state=42)
    # }

    # # Prepare Data
    # X = df.drop(['MedHouseVal','target_class'], axis=1) # using the target_class column for now
    # y = df['MedHouseVal']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # for name, model in models.items():
    #     with mlflow.start_run(run_name=name, nested=True) as run:
    #         model.fit(X_train, y_train)
    #         y_pred = model.predict(X_test)
    #         mlflow.log_param("name", name)


    #         mse = mean_squared_error(y_test, y_pred)
    #         r2 = r2_score(y_test, y_pred)
    #         mlflow.log_metric("mse", mse)
    #         mlflow.log_metric("r2", r2)

    #         input_example = pd.DataFrame(df).sample(1)
    #         mlflow.sklearn.log_model(sk_model=model, artifact_path='model',input_example=input_example)

    #         print(f"{name}: MSE = {mse:.2f}, R2 = {r2:.2f}")


    # from sklearn.model_selection import RandomizedSearchCV
    # import numpy as np

    # with mlflow.start_run(run_name="random_forest_tuning", nested=True) as run:
    #     rf_model = RandomForestRegressor(random_state=42)
    #     param_dist = {
    #             "n_estimators": np.arange(20, 60),
    #             "max_depth": [5, 10, 15, None],
    #             "min_samples_split": np.arange(2, 10),
    #             "min_samples_leaf": np.arange(1, 5)
    #     }
    #     rf_random = RandomizedSearchCV(rf_model, param_dist, n_iter=2, cv=3, random_state=42, scoring='neg_mean_squared_error')
    #     rf_random.fit(X_train, y_train)

    #     best_model = rf_random.best_estimator_
    #     best_params = rf_random.best_params_
    #     y_pred = best_model.predict(X_test)
    #     mse = mean_squared_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred)

    #     mlflow.log_params(best_params)
    #     mlflow.log_metric("mse", mse)
    #     mlflow.log_metric("r2", r2)
    #     mlflow.sklearn.log_model(best_model, "best_model")
    #     print(f"Best Random Forest Model (Tuned): MSE = {mse:.2f}, R2 = {r2:.2f}")