from regression_class import *
from sklearn.linear_model import LinearRegression
import pickle
from regression import predict


def load_y(csv_file, colname):
    return pd.read_csv(csv_file)[colname]


def init_our_model():
    linear_model_revenue = LinearRegression()
    linear_model_votes = LinearRegression()

    split_data("../Data/movies_dataset.csv")
    X = load_data("../Data/training_set.csv")

    y = load_y("../Data/training_set.csv", 'revenue')
    linear_model_revenue.fit(X, y)

    y = load_y("../Data/training_set.csv", 'vote_average')
    linear_model_votes.fit(X, y)

    pickle.dump(linear_model_revenue, "../Data/our_model_revenue.bin")


if __name__ == '__main__':
    init_our_model()
    print(predict("../Data/our_model_revenue.bin"))
