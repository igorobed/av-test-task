from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np


def eval_emb_reduction(model, X, y, msg_embs, reduct="no", scale=False):
    """
    Оценка mse на кросс-валидации при понижении размерности эмбеддинга сообщения коммита
    reduct - ["no", "svd", "pca"] - алгоритм понижения размерности или его отсутствие
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # -1 означает, что мы не применяем понижение размерности эмбеддинга
    if reduct == "no":
        list_n_components = [-1]
    elif reduct == "svd":
        list_n_components = [5, 10, 15, 20, 25, 50, 100, 200, -1]
        reduct_class = TruncatedSVD
    elif reduct == "pca":
        list_n_components = [5, 10, 15, 20, 25, 50, 100, 200, -1]
        reduct_class = PCA

    for n in list_n_components:
        # генератор индексов train и test
        splits = cv.split(X)
        # mse на каждом разбиении
        tmp_lst = []
        for i_train, i_test in splits:
            if n != -1:
                # модель для понижения размерности
                reduct_model = reduct_class(n_components=n)
                msg_embs_train = reduct_model.fit_transform(msg_embs[i_train])
                msg_embs_test = reduct_model.transform(msg_embs[i_test])
            else:
                msg_embs_train = msg_embs[i_train]
                msg_embs_test = msg_embs[i_test]
            X_train, X_test, y_train, y_test = X.loc[i_train], X.loc[i_test], y[i_train], y[i_test]
            
            if scale:
                ss = StandardScaler()
                X_train = ss.fit_transform(X_train)
                X_test = ss.transform(X_test)
            
            X_train_emb = np.concatenate((X_train, msg_embs_train), axis=1)
            X_test_emb = np.concatenate((X_test, msg_embs_test), axis=1)
            model.fit(X_train_emb, y_train)
            preds = model.predict(X_test_emb)
            tmp_lst.append(mean_squared_error(y_test, preds))
        print(f"{n} - {np.mean(tmp_lst)}")


if __name__ == "__main__":
    pass