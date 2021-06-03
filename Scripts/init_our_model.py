from regression_class import *
from sklearn.linear_model import LinearRegression
import pickle
from regression import predict
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def load_y(csv_file, colname):
    return pd.read_csv(csv_file)[colname]


def plot_rmse():
    # Plot rmse for increasing amount of samples
    results = []
    for i in range(1, 101):
        linear_model = LinearRegression()
        X, y_rev, y_votes = basic_load_data("../Data/training_set.csv")
        n = max(round(X.shape[0] * (i / 100)), 1)
        linear_model.fit(X[:n], y_rev[:n])

        X_test, y_test_rev, y_test_votes = basic_load_data("../Data/test_set.csv")
        y_pred = linear_model.predict(X_test)
        results.append(rmse(y_test_rev, y_pred))

    fig = go.Figure(go.Scatter(x=list(range(1, len(results) + 1)), y=results, mode="markers"),
                    layout=go.Layout(title="Model Evaluation Over Increasing Portions Of Training Set",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set")))
    fig.write_image("../Figures/mse.over.training.percentage.png")


def init_our_model():
    linear_model_revenue = LinearRegression()
    linear_model_votes = LinearRegression()

    # split_data("../Data/movies_dataset.csv")
    # X = load_data("../Data/training_set.csv")
    X, y_rev, y_votes = basic_load_data("../Data/training_set.csv")

    # y = load_y("../Data/training_set.csv", 'revenue')
    linear_model_revenue.fit(X, y_rev)
    print("revenue prediction:")
    print(linear_model_revenue.predict(basic_load_data("../Data/test_set.csv")[0]))

    linear_model_votes.fit(X, y_votes)
    print("votes prediction:")
    print(linear_model_votes.predict(basic_load_data("../Data/test_set.csv")[0]))

    # y = load_y("../Data/training_set.csv", 'vote_average')
    # linear_model_votes.fit(X, y_votes)
    outfile = open("../Data/our_model_revenue.bi", 'wb')
    pickle.dump(obj=linear_model_revenue, file=outfile)
    outfile.close()
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
    return linear_model_revenue


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
    plot_rmse()
    model_revenue = init_our_model()
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
