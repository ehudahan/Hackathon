import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from ast import literal_eval


def load_data(X):
    """
    get csv file path
    make all preproccicing
    return pandas data frame
    :param csv_file:
    :return:
    """
    df = pd.read_csv("../Data/training_set.csv", converters={'belongs_to_collection': literal_converter})
    # t = [list(pd.json_normalize(c)['id'].values) for c in df['belongs_to_collection'][:20]]
    df['id_collection'] = pd.DataFrame([list(pd.json_normalize(c)['id'].values) for c in df['belongs_to_collection']])
    df['is_in_collection'] = df['id_collection'].notna().astype('int')
    # df['is_in_collection'] =
    print(df.is_in_collection)

    df = df.drop(['belongs_to_collection'])

    X = pd.concat([X.drop(['belongs_to_collection'], axis=1),
               X.belongs_to_collection.apply(X.Series)], axis=1)
    X.drop(["id", "belongs_to_collection", "genre", "homepage", "original_language", "original_title", "overview",
            "production_companies", "production_countries", "spoken_languages", "status", "tagline", "title",
            "keywords", "cast", "crew"], axis=1)

    # json(df.belongs_to_collection)
    # df2 = pd.json_normalize(df.belongs_to_collection)
    # df = df.join(pd.json_normalize(df.belongs_to_collection))
    return X


# df = pd.read_csv("movies_dataset.csv")
# print(type(json.loads(df.belongs_to_collection)))

# print(df.belongs_to_collection[type(df.belongs_to_collection]))
# print(json.load(df.belongs_to_collection[df.belongs_to_collection != ""]))
# pd.concat([df.drop(['belongs_to_collection'], axis=1),
#            df.belongs_to_collection.apply(pd.Series)], axis=1)

# df2 = pd.json_normalize(df.belongs_to_collection)
# print(df2.head())

def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return {'id': pd.NA} if (val == "") or (val == "[]") else literal_eval(val)


class Reg():
    """
    This is a regression class model
    """
    def fit(self):
        pass

    def predict(self):
        pass

    def score(self):
        pass


if __name__ == '__main__':
    df = pd.read_csv("../Data/training_set.csv", converters={'belongs_to_collection': literal_converter,
                                                             'genres': literal_converter})
    # t = [list(pd.json_normalize(c)['id'].values) for c in df['belongs_to_collection'][:20]]
    df['id_collection'] = pd.DataFrame([list(pd.json_normalize(c)['id'].values) for c in df['belongs_to_collection']])
    df['is_in_collection'] = df['id_collection'].notna().astype('int')

    df['id_genres'] = [pd.json_normalize(c)['id'].to_list() for c in df['genres']]
    print(df['id_genres'])





    # new_df = pd.concat([pd.DataFrame(pd.json_normalize(g)) for g in df['genres']], ignore_index=True)
    # df2 = df[:3]
    # t = list(pd.json_normalize(g)['name'].values) for g in df['genres'][:6]]
    # ['name'].values

    # df = df.join(pd.json_normalize(df.Information))

    # df.drop(columns=['Information'], inplace=True)
    # pd.json_normalize(genre) for genre in df.genres
    # json_normalize(x)) for x in df['json']
        # .drop(columns=['genres'])
    # y_rank = df["vote_average"]
    # y_revenue = df["revenue"]
    # X = df.drop(["revenue", "vote_average"], axis=1)
    # new_X = load_data(X)
    # print(new_X.head)

