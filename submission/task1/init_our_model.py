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
    # results_lasso = []
    X = load_data("../../Data/movies_dataset_part2.csv")
    y_test = X['revenue']
    y_test_vote = X['vote_average']
    X_test = X.drop['revenue', 'vote_average']
    for i in range(1, 101):
        linear_model = LinearRegression()
        # lasso_model = Lasso(alpha=1.0)
        n = max(round(X.shape[0] * (i / 100)), 1)

        X_test, y_test_rev, y_test_votes = basic_load_data("../Data/test_set.csv")
        results_lin.append(rmse(y_test_rev, linear_model.predict(X_test)))
        # results_lasso.append(rmse(y_test_rev, lasso_model.predict(X_test)))

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


def split_data_by_zero_rev(filename):
    """
    Load movies prices dataset split the data
    :param filename: Path to movies prices dataset
    :return: Training_set = 3/4, Test_set = 1/4, with respect to the revenue field
    """
    df = pd.read_csv(filename).drop_duplicates()
    df0 = df[df['revenue'] == 0]
    df1 = df[df['revenue'] != 0]
    return df0, df1


def init_our_model():
    X = pd.read_csv("../../Data/Data_after_preproccecing.csv")
    y_rev = X['revenue']
    y_votes = X['vote_average']
    X = X.drop(['revenue', 'vote_average'], axis=1)

    model_list = [LinearRegression(), LinearRegression()]
    print("init X.shape", X.shape)
    print("init columns:", X.columns)
    model_list[0].fit(X, y_rev)
    model_list[1].fit(X, y_votes)

    outfile = open("our_models.bi", 'wb')
    pickle.dump(obj=model_list, file=outfile)
    outfile.close()


def cor_mat(X):
    df = pd.DataFrame(X)
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    plt.savefig("../../Figures/corr_mat.png")


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
    # load_data("../../Data/movies_dataset.csv", True)
    # cor_mat(pd.read_csv("../../Data/Data_after_preproccecing.csv"))
    # plot_rmse()
    init_our_model()