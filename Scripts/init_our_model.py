from parsing import *
from sklearn.linear_model import LinearRegression, Lasso
import pickle
from regression import predict
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sn
import matplotlib.pyplot as plt


def load_y(csv_file, colname):
    return pd.read_csv(csv_file)[colname]


def plot_rmse():
    # Plot rmse for increasing amount of samples
    results_lin = []
    results_lasso = []
    for i in range(1, 101):
        linear_model = LinearRegression()
        lasso_model = Lasso(alpha=1.0)
        X, y_rev, y_votes = basic_load_data("../Data/training_set.csv")
        n = max(round(X.shape[0] * (i / 100)), 1)
        linear_model.fit(X[:n], y_rev[:n])
        lasso_model.fit(X[:n], y_rev[:n])

        X_test, y_test_rev, y_test_votes = basic_load_data("../Data/test_set.csv")
        results_lin.append(rmse(y_test_rev, linear_model.predict(X_test)))
        results_lasso.append(rmse(y_test_rev, lasso_model.predict(X_test)))

    fig = go.Figure(go.Scatter(x=list(range(1, len(results_lin) + 1)), y=results_lin, mode="markers"),
                    layout=go.Layout(title="Model Evaluation Over Increasing Portions Of Training Set",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set")))
    fig.write_image("../Figures/mse.over.training.percentage.lin.png")

    fig = go.Figure(go.Scatter(x=list(range(1, len(results_lasso) + 1)), y=results_lin, mode="markers"),
                    layout=go.Layout(title="Model Evaluation Over Increasing Portions Of Training Set",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set")))
    fig.write_image("../Figures/mse.over.training.percentage.lasso.png")


def filter_by_corr_mat(X, y):
    """
    :param X: data frame
    :param y: vector
    :return: return list of choosen variables
    """
    mat = pd.concat([X, y], axis=1).corr()
    # print(mat.columns[mat['revenue'] > 0.2])


def init_our_model():
    linear_model_revenue = LinearRegression()
    linear_model_votes = LinearRegression()
    # lasso_model_revenue = Lasso(alpha=1.0)

    # split_data("../Data/movies_dataset.csv")
    X = load_data("../Data/training_set.csv")
    print(X.columns)
    y_rev = X['revenue']
    y_votes = X['vote_average']
    X = X.drop(['revenue', 'vote_average'], axis=1)
    # y_votes = pd.read_csv("../Data/training_set.csv")['vote_average']
    # X_filtered = filter_by_corr_mat(X, y)
    # cor_mat(X_filtered)

    # cor_mat(X[['budget', 'vote_count', "runtime"]])
    # cor_mat(pd.read_csv("../Data/movies_dataset.csv"))


    # y = load_y("../Data/training_set.csv", 'revenue')
    linear_model_revenue.fit(X, y_rev)
    # print("revenue prediction:")
    # print(linear_model_revenue.predict(basic_load_data("../Data/test_set.csv")[0]))
    outfile = open("../Data/our_model_revenue.bi", 'wb')
    pickle.dump(obj=linear_model_revenue, file=outfile)
    outfile.close()
    #
    linear_model_votes.fit(X, y_votes)
    # print("votes prediction:")
    # print(linear_model_votes.predict(basic_load_data("../Data/test_set.csv")[0]))
    outfile = open("../Data/our_model_votes.bi", 'wb')
    pickle.dump(obj=linear_model_votes, file=outfile)
    outfile.close()

    # y = load_y("../Data/training_set.csv", 'vote_average')
    # linear_model_votes.fit(X, y_votes)
    # plot_singular_values(X)
    # resample with replacement each row
    # n_boots = 10
    # weights = np.array((n_boots, X.shape[1]))
    # n_points = X.shape[0]
    # plt.figure()
    # for i in range(n_boots):
    #     # sample the rows, same size, with replacement
    #     X, y_rev, y_votes = basic_load_data("../Data/training_set.csv")
    #     X_train = X.sample(n=n_points, replace=True)
    #     y_train = y_rev[X.index]
    #     # fit a linear regression
    #     linear_model = LinearRegression()
    #     results_temp = linear_model.fit(X_train, y_train)
    #
    #     # append coefficients
    #     weights[i] = results_temp
    #
    #     # plot a greyed out line
    #     # y_pred_temp = linear_model.predict(basic_load_data("../Data/test_set.csv"))
    #     # linear_model.coef_ = np.mean(weights[:i, :])
    #
    # lasso_model_revenue.fit(X, y_rev)
    # return linear_model_revenue


def cor_mat(X):
    df = pd.DataFrame(X)
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix[abs(corrMatrix) > 0.2], annot=True)
    plt.show()
    plt.savefig("../Figures/corr_mat.png")


def rmse(y, y_pred):
    """
    Calculate the MSE given the true- and prediction- vectors
    :param y: The true response vector
    :param y_pred: The predicted response vector
    :return: MSE of the prediction
    """
    return np.sqrt(np.mean((y - y_pred) ** 2))


def plot_singular_values(X):
    """
    Given a design matrix X, plot the singular values of all non-categorical features
    :param X: The design matrix to use
    """
    sv = np.linalg.svd(X, compute_uv=False)
    fig = go.Figure(go.Scatter(x=X.columns, y=sv, mode='lines+markers'),
                    layout=go.Layout(title="Scree Plot of Design Matrix Singular Values",
                                     xaxis=dict(title=""), yaxis=dict(title="Singular Values")))
    fig.write_image("../Figures/singular.values.scree.plot.png")


def feature_evaluation(X, y):
    for f in X:
        rho = np.cov(X[f], y)[0, 1] / (np.std(X[f]) * np.std(y))

        fig = px.scatter(pd.DataFrame({'x': X[f], 'y': y}), x="x", y="y", trendline="ols",
                         title=f"Correlation Between {f} Values and Response <br>Pearson Correlation {rho}",
                         labels={"x": f"{f} Values", "y": "Response Values"})
        fig.write_image("../Figures/pearson.correlation.%s.png" % f)


if __name__ == '__main__':
    # plot_rmse()
    init_our_model()
    # X = load_data("../Data/test_set.csv")
    # y = load_y("../Data/test_set.csv", "revenue")


    #
    # for i in range(1, 101):
    #     n = max(round(len(train[1]) * (i/100)), 1)
    #     results.append(_fit_and_test((train[0][:n], train[1][:n]), test))
    #
    # fig = go.Figure(go.Scatter(x=list(range(1, len(results)+1)), y=results, mode="markers"),
    #                 layout=go.Layout(title="Model Evaluation Over Increasing Portions Of Training Set",
    #                                  xaxis=dict(title="Percentage of Training Set"),
    #                                  yaxis=dict(title="MSE Over Test Set")))
    # fig.write_image("mse.over.training.percentage.png")
