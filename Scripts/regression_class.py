import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from ast import literal_eval


def split_data(filename):
    """
    Load movies prices dataset split the data
    :param filename: Path to movies prices dataset
    :return: Training_set = 3/4, Test_set = 1/4, with respect to the revenue field
    """
    df = pd.read_csv(filename).drop_duplicates()

    df_sorted_by_revenue_value = df.sort_values(by=['revenue'])

    # splitting the data : training_set = 3/4, test_set = 1/4, with respect to the revenue field
    test_set = df_sorted_by_revenue_value[::4]
    test_rows = test_set["id"]
    training_set = df_sorted_by_revenue_value.drop(index=test_rows - 1)

    # adding mean values of the following fields: original_language,vote_average,vote_count,production_companies,
    pd.DataFrame.to_csv(training_set)

    # write CSVs of the training set and test set
    training_set.to_csv('../Data/training_set.csv', header=True, encoding='utf-8', index=False)
    test_set.to_csv('../Data/test_set.csv', header=True, encoding='utf-8', index=False)


def literal_converter_id(val):
    # replace first val with '' or some other null identifier if required
    return {'id': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def literal_converter_iso(val):
    return {'iso_3166_1': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def literal_converter_lan(val):
    return {'iso_639_1': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def literal_converter_crew(val):
    return {'gender': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def columns_to_drop():
    return ["id", "belongs_to_collection", "genres", "homepage", "original_language", "original_title", "overview",
            "production_companies", "release_date", "production_countries", "spoken_languages", "status", "tagline", "title",
            "keywords", "cast", "crew"]


def load_data(csv_file):
    """
    get csv file path (training or test set)
    make all preproccicing
    return pandas data frame
    :param csv_file:
    :return:
    """
    columns_to_convert = {'belongs_to_collection': literal_converter_id,
                          'genres': literal_converter_id,
                          'production_companies': literal_converter_id,
                          'keywords': literal_converter_id,
                          'production_countries': literal_converter_iso,
                          'spoken_languages': literal_converter_lan,
                          'cast': literal_converter_crew,
                          'crew': literal_converter_crew
                          }
    df = pd.read_csv(csv_file, converters=columns_to_convert)
    id_list = ['belongs_to_collection', 'genres', 'production_companies', 'keywords']
    iso_list = ['production_countries', 'spoken_languages']
    crew_list = ['cast', 'crew']
    for colname in id_list:
        df['id_' + str(colname)] = [pd.json_normalize(c)['id'].to_list() for c in df[colname]]
    # for colname in id_list:
    #     df['id_' + str(colname)] = [pd.json_normalize(c)['id'].to_list() for c in df[colname]]
    # for colname in id_list:
    #     df['id_' + str(colname)] = [pd.json_normalize(c)['id'].to_list() for c in df[colname]]

    # df = pd.get_dummies(df, columns=['original_language'])

    return df.drop(columns_to_drop(), axis=1)


def basic_load_data(csv_file):
    df = pd.read_csv(csv_file).dropna()
    return df[['budget', 'vote_count', 'runtime']], df['revenue'], df['vote_average']



if __name__ == '__main__':
    X = load_data("../Data/training_set.csv")
