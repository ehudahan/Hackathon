import pandas as pd
import json


def load_data(csv_file):
    """
    get csv file path
    make all preproccicing
    return pandas data frame
    :param csv_file:
    :return:
    """
    df = pd.read_csv(csv_file)
    df = pd.concat([df.drop(['belongs_to_collection'], axis=1),
               df.belongs_to_collection.apply(pd.Series)], axis=1)
    # json(df.belongs_to_collection)
    # df2 = pd.json_normalize(df.belongs_to_collection)
    # df = df.join(pd.json_normalize(df.belongs_to_collection))
    return df


df = pd.read_csv("movies_dataset.csv")
print(type(json.loads(df.belongs_to_collection)))
# print(df.belongs_to_collection[type(df.belongs_to_collection]))
# print(json.load(df.belongs_to_collection[df.belongs_to_collection != ""]))
# pd.concat([df.drop(['belongs_to_collection'], axis=1),
#            df.belongs_to_collection.apply(pd.Series)], axis=1)

# df2 = pd.json_normalize(df.belongs_to_collection)
# print(df2.head())


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
    df = load_data("movies_dataset.csv")
    print(df.head)