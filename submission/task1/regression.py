################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

from parsing import *
import pickle


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    X = load_data(csv_file)
    infile = open("our_models.bi", 'rb')
    models_list = pickle.load(infile)
    infile.close()
    return list(models_list[0].predict(X)), list(models_list[1].predict(X))


if __name__ == '__main__':
    predict("../../Data/movies_dataset_part2_test.csv")