import dill
import pickle

import pandas as pd

from sklearn.impute import SimpleImputer
from datetime import datetime
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


def del_columns(df):
    columns_to_drop = [
        'device_model',
        'utm_keyword',
        'device_os',
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'device_screen_resolution',
        'geo_country',
        'geo_city',
        'utm_adcontent'
    ]
    return df.drop(columns_to_drop, axis=1)


def full_date(df):
    import pandas
    df['full_date'] = pandas.to_datetime(
        df['visit_date'].astype(str) + ' '
        + df['visit_time'].astype(str)
    )
    df['hour'] = df.apply(
        lambda x: x.full_date.time().hour, axis=1)
    df['month'] = df.apply(
        lambda x: x.full_date.date().month, axis=1)
    return df



def device_screen_resol(x):
    try:
        a = int(x.split('x')[0])
        b = int(x.split('x')[1])
        return int(a * b)
    except:
        return 0


def data_device_screen_resol(df):
    def device_screen_resol(x):
        try:
            a = int(x.split('x')[0])
            b = int(x.split('x')[1])
            return int(a * b)
        except:
            return 0

    df["dev_screen_resol_"] = df["device_screen_resolution"]
    df["dev_screen_resol_"] = df["dev_screen_resol_"].apply(device_screen_resol)
    # заменяем not set на моду
    df.dev_screen_resol_.replace('not set', (df.dev_screen_resol_.mode()[0]), inplace=True)
    return df



def lat_long(df):
    # Укажи верный путь к словарям
    import pickle
    with open('data/lat_dict.pkl', 'rb') as file:
        lat_dict = pickle.load(file)
    with open('data/long_dict.pkl', 'rb') as file:
        long_dict = pickle.load(file)

    def lat(x):
        b = lat_dict[x]
        return b

    def long(x):
        b = long_dict[x]
        return b

    df['geo_city_country'] = df.apply(
        lambda x: ''.join((x['geo_city'], x['geo_country'])), axis=1)

    df['lat'] = df.apply(lambda x: (lat(x['geo_city_country'])), axis=1)
    df['long'] = df.apply(lambda x: (long(x['geo_city_country'])), axis=1)
    return df


def main():
    import pandas
    import dill
    from datetime import datetime
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import make_column_selector, ColumnTransformer
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression


    df = pandas.read_csv('data/session_to_modeling.csv', low_memory=False)

    x = df.drop('target', axis=1)
    y = df['target']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        # ('imputer',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    from sklearn.impute import SimpleImputer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('full_date', FunctionTransformer(full_date)),
        ('data_device_screen_resol', FunctionTransformer(data_device_screen_resol)),
        ('feature_creator', FunctionTransformer(lat_long)),
        ('del_columns', FunctionTransformer(del_columns)),
        ('column_transformer', column_transformer)
    ])

    models = [
        LogisticRegression(class_weight='balanced'),
        RandomForestClassifier(),
        MLPClassifier()
    ]

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        roc_auc = cross_val_score(pipe, x, y, cv=2, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {roc_auc.mean():.4f}')
        if roc_auc.mean() > best_score:
            best_score = roc_auc.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(x, y)

    with open('CR_pred.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'CR prediction model',
                'author': 'VALIEV Ruslan',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()