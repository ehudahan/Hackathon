import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from ast import literal_eval


def literal_converter_id(val):
    # replace first val with '' or some other null identifier if required
    return {'id': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def literal_converter_iso(val):
    return {'iso_3166_1': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def literal_converter_lan(val):
    return {'iso_639_1': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def literal_converter_crew(val):
    return {'gender': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def load_data(csv_file):
    """
    get csv file path
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
    columns_to_drop = ["id", "belongs_to_collection", "genres", "homepage", "original_language", "original_title", "overview",
            "production_companies", "production_countries", "spoken_languages", "status", "tagline", "title",
            "keywords", "cast", "crew"]

    return df.drop(columns_to_drop, axis=1)


if __name__ == '__main__':
    load_data("../Data/training_set.csv")

