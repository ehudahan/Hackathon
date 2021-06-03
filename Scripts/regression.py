################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

from regression_class import *


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    X = preproccecing(csv_file)
    Reg.predict(X, y)
    # reg = LinearRegression().fit(X, y)

    pass
