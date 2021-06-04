import pandas as pd
from ast import literal_eval
import pickle
import re

LANGUAGE_NAMES = ['en', 'fr', 'hi', 'es', 'ja', 'ru', 'it', 'ko', 'ta', 'zh']
POP_GENRES = ["genre_10749", "genre_18", "genre_35", "genre_53", "genre_80", "genre_18"]
POP_CREW = ["crew_4185"]
COLS_DROP = ["id", "belongs_to_collection", "genres", "homepage", "original_language", "original_title", "overview",
             "production_companies", "production_countries", "release_date", "spoken_languages", "tagline", "title",
             "keywords", "cast", "crew"]
COLS_DROP2 = ["id_genres",
              "id_production_companies", "id_keywords", "id_countries",
              "id_lan", "pop_words", "id_collection"]
id_list = ['genres', 'production_companies', 'keywords']
iso_list = ['production_countries', 'spoken_languages']
crew_list = ['cast', 'crew']

pd.options.mode.chained_assignment = None


def literal_converter_id(val):
    # replace first val with '' or some other null identifier if required
    return {'id': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def literal_converter_iso(val):
    return {'iso_3166_1': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def literal_converter_lan(val):
    return {'iso_639_1': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def literal_converter_crew(val):
    return {'id': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


def time_variable(df):
    """get a df with release_date time-variable as 16/04/1992 and split to years and 12 months"""
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['year'] = df['release_date'].dt.year
    df['month'] = df['release_date'].dt.month
    return df


def create_pop_words_list(words_set):
    """words_set - lists of words
    count number of known words in movie description
    Known word are the 10% percent of incident words in the data set
    @return list of first 10% highest words in the data set"""
    # print(words_set[:51])
    words_amounts = dict()
    for i in range(len(words_set)):
        try:
            # print("i=", i, ":", words_set[i])
            sentence = words_set[i].split(' ')
            for j in range(len(sentence)):
                word = sentence[j].lower()
                if word in words_amounts:
                    words_amounts[word] += 1
                else:
                    words_amounts[word] = 1
        except:
            pass
    pop_words = sorted(words_amounts, key=words_amounts.get, reverse=True)[:1000]
    outfile = open("Data/pop_words.bi", 'wb')
    pickle.dump(obj=pop_words, file=outfile)
    outfile.close()


def words_dict(words_set):
    infile = open("pop_words.bi", 'rb')
    pop_words = pickle.load(infile)
    infile.close()

    res = []
    for i in range(len(words_set)):
        l = []
        try:
            sentence = words_set[i].split(' ')
            for j in range(len(sentence)):
                word = sentence[j].lower()
                if word in pop_words:
                    l.append(word)
        except:
            pass
        res.append(l)
    return res


def parse_jsons(df):
    '''
    parse json column to relevant columns

    '''
    df['id_collection'] = pd.DataFrame([list(pd.json_normalize(c)['id'].values) for c in df['belongs_to_collection']])
    id_list = ['genres', 'production_companies', 'keywords']
    for colname in id_list:
        df['id_' + str(colname)] = [str(pd.json_normalize(c)['id'].tolist())[1:-1] for c in df[colname]]
    df['id_countries'] = [str(pd.json_normalize(c)['iso_3166_1'].to_list())[1:-1] for c in df['production_countries']]
    df['id_lan'] = [str(pd.json_normalize(c)['iso_639_1'].to_list())[1:-1] for c in df['spoken_languages']]
    # df['cast_ids'] = [str(pd.json_normalize(c)['id'].to_list())[1:-1] for c in df['cast']]
    # df['crew_ids'] = [str(pd.json_normalize(c)['id'].to_list())[1:-1] for c in df['crew']]

    df['com_website'] = df.homepage.apply(lambda x: 1 if re.match(r".com.", str(x)) else 0)
    df['title_len'] = df.original_title.apply(lambda x: len(str(x)))
    # create_pop_words_list(df.overview) # Need to run only once
    df['pop_words'] = words_dict(df.overview)
    return df


def add_dummies(df):
    df['status'] = df['status'].apply(lambda x: 1 if x == "Released" else 0)
    df['is_in_collection'] = df['id_collection'].notna().astype('int')
    for l in LANGUAGE_NAMES:
        df[l + "_language"] = (df["original_language"] == l).astype('int')
    temp = df['id_genres'].str.get_dummies(sep=',').rename(lambda x: 'genre_' + x, axis='columns')
    for g in POP_GENRES:
        if g in temp.columns:
            df[g] = temp[g]
        else:
            df[g] = 0

    # temp_crew = df['crew_ids'].str.get_dummies(sep=',').rename(lambda x: 'crew_' + x, axis='columns')
    # temp_cast = df['cast_ids'].str.get_dummies(sep=',').rename(lambda x: 'cast_' + x, axis='columns')
    # df = pd.concat(df, temp_crew)
    # for c in POP_CREW:
    #     if c in temp_crew.columns:
    #         df[c] = temp_crew[c]
    #     else:
    #         df[c] = 0

    # df = pd.concat([df, df['status'].str.get_dummies(sep=',').rename(lambda x: 'status_' + x, axis='columns')], axis=1)
    # .drop(["status_<NA>"], axis=1)
    return df


def fill_missing(df, avg):
    for col in avg:
        df[col].replace(to_replace=0, value=avg[col])


def find_avg(df):
    """ find avg or most common item in column or """
    avg = dict()

    l = ['budget', 'vote_count', 'runtime', 'month', 'year']

    for i in l:
        avg[i] = ((df[df[i] != 0])[i].mean(skipna=True))
        if i != 'vote_average':
            avg[i] = int(avg[i])

    outfile = open("avg_dict.bi", 'wb')
    pickle.dump(obj=avg, file=outfile)
    outfile.close()


def filter_training_data(X):
    """"
    filter not relevant columns for training
    """
    X = X[X['revenue'] != 0]
    X = X[X['budget'] != 0]
    return X


def zero_nan_carring(df, write_dict):
    """
    load the avg dictionaty and fill the values
    :param df:
    :return: df
    """
    if write_dict:
        find_avg(df)
    infile = open("avg_dict.bi", 'rb')
    avg_dict = pickle.load(infile)
    infile.close()

    df = df.fillna(0)

    fill_missing(df, avg_dict)
    return df


def add_missing_columns(df):
    """
    add missing columns to the data
    :param df:
    :return: df
    """
    column_list = pd.read_csv("model_columns.csv").columns
    print(column_list)
    for col in column_list:
        if col not in df.columns:
            df[col] = 0
    return df


def normalized(df, write_dict):
    """
    Normalise and standartized variables
    :param df:
    :return:
    """
    if write_dict:
        std_dict = {}
        for col in df.columns:
            if max(df[col]) > 10:
                std_dict[col] = df[col].mean(), df[col].std()
                mu, std = std_dict[col]
                df[col] = (df[col] - mu) / std
        outfile = open("std_dict.bi", 'wb')
        pickle.dump(obj=std_dict, file=outfile)
        outfile.close()
    else:
        infile = open("std_dict.bi", 'rb')
        std_dict = pickle.load(infile)
        infile.close()
        for col in std_dict.keys():
            mu, std = std_dict[col]
            df[col] = (df[col] - mu) / std
    return df




def load_data(csv_file, train_run=False):
    """
    get csv file path
    make all preprocessing
    return pandas data frame
    :param csv_file:
    :return:
    """
    df = pd.read_csv(csv_file, converters={'belongs_to_collection': literal_converter_id,
                                           'genres': literal_converter_id,
                                           'production_companies': literal_converter_id,
                                           'keywords': literal_converter_id,
                                           'production_countries': literal_converter_iso,
                                           'spoken_languages': literal_converter_lan,
                                           'cast': literal_converter_crew,
                                           'crew': literal_converter_crew
                                           })
    if train_run:
        df = filter_training_data(df)
    df = parse_jsons(df)
    df = add_dummies(df)
    df = time_variable(df)
    df = zero_nan_carring(df, train_run)

    df = df.drop(COLS_DROP + COLS_DROP2, axis=1)
    # df = normalized(df, train_run)

    if train_run:
        df.to_csv("../../Data/Data_after_preproccecing.csv", index=False)
        df[:1].to_csv("model_columns.csv", index=False)
    else:
        df = add_missing_columns(df)
        if 'revenue' in df.columns:
            df = df.drop(['revenue'], axis=1)
        if 'vote_average' in df.columns:
            df = df.drop(['vote_average'], axis=1)

    return df