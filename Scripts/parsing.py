import pandas as pd
from ast import literal_eval
import pickle
import re

LANGUAGE_NAMES = ['en', 'fr', 'hi', 'es', 'ja', 'ru', 'it', 'ko', 'ta', 'zh']
COLS_DROP = ["id", "belongs_to_collection", "genres", "homepage", "original_language", "original_title", "overview",
             "production_companies", "production_countries", "release_date", "spoken_languages", "tagline", "title",
             "keywords", "cast", "crew"]

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
    df['is_in_collection'] = df['id_collection'].notna().astype('int')
    for l in LANGUAGE_NAMES:
        df[l + "_language"] = (df["original_language"] == l).astype('int')
    df = pd.concat([df, df['id_genres'].str.get_dummies(sep=',').rename(lambda x: 'genre_' + x, axis='columns')],
                   axis=1)
    # df = pd.concat([df, df['crew_names'].str.get_dummies(sep=',').rename(lambda x: 'crew_' + x, axis='columns')],
    #                axis=1)
    df = pd.concat([df, df['status'].str.get_dummies(sep=',').rename(lambda x: 'status_' + x, axis='columns')], axis=1)
    # .drop(["status_<NA>"], axis=1)
    return df


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

    avg['original_language'] = 'en'
    return avg


def fill_missing(df, avg):
    for col in avg:
        for i in range(len(df)):
            if df[col][i] == 0:
                df[col][i] = avg[col]


def load_data(csv_file):
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

    df = df.drop(COLS_DROP, axis=1)
    df.to_csv("try.csv")
    avg = find_avg(df)
    return df


# if __name__ == '__main__':
#     # split_data("Data/movies_dataset.csv")
#     load_data("Data/training_set.csv")

####################### OLD ########
if 0:
    import pandas as pd
    import json
    from sklearn.linear_model import LinearRegression
    from ast import literal_eval
    import seaborn as sn
    import matplotlib.pyplot as plt
    import re
    import pickle


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
        return {'name': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


    def columns_to_drop():
        return ["id", "belongs_to_collection", "genres", "homepage", "original_language", "original_title", "overview",
                "production_companies", "release_date", "production_countries", "spoken_languages", "status", "tagline", "title",
                "keywords", "cast", "crew"]


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
        outfile = open("../Data/pop_words.bi", 'wb')
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


    def load_data(csv_file):
        """
        get csv file path (training or test set)
        make all preproccecing
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

        for colname in id_list:
            df['id_' + str(colname)] = [pd.json_normalize(c)['id'].to_list() for c in df[colname]]
        df['id_countries'] = [pd.json_normalize(c)['iso_3166_1'].to_list() for c in df['production_countries']]
        df['id_lan'] = [pd.json_normalize(c)['iso_639_1'].to_list() for c in df['spoken_languages']]
        df['cast_names'] = [pd.json_normalize(c)['name'].to_list() for c in df['cast']]
        df['crew_names'] = [pd.json_normalize(c)['name'].to_list() for c in df['crew']]

        df['com_website'] = df.homepage.apply(lambda x: 1 if re.match(r".*com.*", str(x)) else 0)
        df['title_len'] = df.original_title.apply(lambda x: len(str(x)))
        # create_pop_words_list(df.overview) # Need to run only once
        df['pop_words'] = words_dict(df.overview)
        print(df['pop_words'])


        # df = pd.get_dummies(df, columns=['original_language'])

        return df.drop(columns_to_drop(), axis=1)


    def basic_load_data(csv_file):
        df = pd.read_csv(csv_file).dropna()
        return df[['budget', 'vote_count', 'runtime']], df['revenue'], df['vote_average']



    if __name__ == '__main__':
        X = load_data("../Data/training_set.csv")
        # cor_mat(X[['budget', 'vote_count', "runtime"]])
        # cor_mat(pd.read_csv("../Data/movies_dataset.csv"))
