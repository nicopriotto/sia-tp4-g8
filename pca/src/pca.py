import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data(path, country_column, feature_columns):
    df = pd.read_csv(path)
    countries = df[country_column].tolist()
    X = df[feature_columns].to_numpy(dtype=float)
    return df, X, countries


def standardize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def run_pca(X_std, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    return pca


def autovectores_df(pca, feature_names):
    columns = [f"PC{i + 1}" for i in range(pca.n_components_)]
    return pd.DataFrame(pca.components_.T, index=feature_names, columns=columns)


def scores_df(pca, X_std, countries):
    columns = [f"PC{i + 1}" for i in range(pca.n_components_)]
    return pd.DataFrame(pca.transform(X_std), index=countries, columns=columns)


def autovalores_summary(pca):
    return pd.DataFrame({
        "autovalor": pca.explained_variance_,
        "proporcion": pca.explained_variance_ratio_,
        "acumulada": np.cumsum(pca.explained_variance_ratio_),
    }, index=[f"PC{i + 1}" for i in range(pca.n_components_)])
