################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

from regression_class import *
import pickle


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    X = load_data(csv_file)
    infile = open("../Data/our_model_revenue.bi", 'rb')
    model = pickle.load(infile)
    infile.close()
    return model.predict(X)


def predict_revenue(w):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    y_pred = w.predict(X)
    print('predicted response:', y_pred, sep='\n')
    return y_pred