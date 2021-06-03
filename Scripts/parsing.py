import pandas as pd
from ast import literal_eval
import pickle
import re

LANGUAGE_NAMES = ['en', 'fr', 'hi', 'es', 'ja', 'ru', 'it', 'ko', 'ta', 'zh']
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
    return {'name': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


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
    infile = open("../Data/pop_words.bi", 'rb')
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

    '''
    df['id_collection'] = pd.DataFrame([list(pd.json_normalize(c)['id'].values) for c in df['belongs_to_collection']])
    id_list = ['genres', 'production_companies', 'keywords']
    iso_list = ['production_countries', 'spoken_languages']
    crew_list = ['cast', 'crew']
    for colname in id_list:
        df['id_' + str(colname)] = [str(pd.json_normalize(c)['id'].tolist())[1:-1] for c in df[colname]]
    df['id_countries'] = [str(pd.json_normalize(c)['iso_3166_1'].to_list())[1:-1] for c in df['production_countries']]
    df['id_lan'] = [str(pd.json_normalize(c)['iso_639_1'].to_list())[1:-1] for c in df['spoken_languages']]
    # df['cast_names'] = [str(pd.json_normalize(c)['name'].to_list())[1:-1] for c in df['cast']]
    # df['crew_names'] = [str(pd.json_normalize(c)['name'].to_list())[1:-1] for c in df['crew']]

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
    df = pd.concat([df, df['id_genres'].str.get_dummies(sep=',').rename(lambda x: 'genre_' + x, axis='columns')],
                   axis=1)
    # df = pd.concat([df, df['crew_names'].str.get_dummies(sep=',').rename(lambda x: 'crew_' + x, axis='columns')],
    #                axis=1)
    # df = pd.concat([df, df['status'].str.get_dummies(sep=',').rename(lambda x: 'status_' + x, axis='columns')], axis=1)
    # .drop(["status_<NA>"], axis=1)
    return df


def fill_missing(df, avg):
    for col in avg:
        for i in range(len(df)):
            if df[col][i] == 0:
                df[col][i] = avg[col]


def find_avg(df):
    """ find avg or most common item in column or """
    avg = dict()

    l = ['revenue', 'budget', 'vote_average', 'vote_count', 'runtime']

    for i in l:
        avg[i] = ((df[df[i] != 0])[i].mean())
        if i != 'vote_average':
            avg[i] = int(avg[i])

    for kw in ['month', 'year']:
        avg[kw] = int(((df[df[kw] != 0])[kw].mean()))
    outfile = open("../Data/avg_dict.bi", 'wb')
    pickle.dump(obj=avg, file=outfile)
    outfile.close()


def zero_nan_carring(df, load_dict=True):
    """
    load the avg dictionaty and fill the values
    :param df:
    :return: df
    """
    if not load_dict:
        find_avg(df)
    infile = open("../Data/avg_dict.bi", 'rb')
    avg_dict = pickle.load(infile)
    infile.close()

    df.fillna(0)

    for col in avg_dict:
        for i in range(len(df)):
            if df[col][i] == 0 or df[col][i] == "":
                df[col][i] = avg_dict[col]
    return df



def load_data(csv_file, save_csv=False):
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
    df = parse_jsons(df)
    df = add_dummies(df)
    df = time_variable(df)
    df = zero_nan_carring(df, not save_csv)

    df = df.drop(COLS_DROP + COLS_DROP2, axis=1)
    if save_csv:
        df.to_csv("../Data/Data_after_preproccecing.csv")

    return df