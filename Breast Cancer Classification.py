## Dataset

# The dataset used in this project contains information about 569 instances of cancer diagnosis, with class distribution
# as follows: 212 Malignant and 357 Benign. The features are linearly separable, and the dataset includes 30 different
# features related to the characteristics of cell nuclei present in the digitized image of a fine needle aspirate (FNA)
# of a breast mass.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1500)

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                         columns=np.append(cancer['feature_names'], ['target']))


def check_df(dataframe, head=5):
    print("#################### Shape ####################")
    print(dataframe.shape)
    print("#################### Types ####################")
    print(dataframe.dtypes)
    print("#################### Num of Unique ####################")
    print(dataframe.nunique())  # "dataframe.nunique(dropna=False)" yazarsak null'larıda veriyor.
    print("#################### Head ####################")
    print(dataframe.head(head))
    print("#################### Tail ####################")
    print(dataframe.tail(head))
    print("#################### NA ####################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ####################")
    print(dataframe.describe([0.01, 0.05, 0.75, 0.90, 0.95, 0.99]).T)


check_df(df_cancer)


def grab_col_names(dataframe, cat_th=16, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken
        sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df_cancer)
df_cancer.groupby("target").agg({col: lambda x: x.mean() for col in num_cols})


## Data Visualization

sns.pairplot(df_cancer,
             hue="target",
             vars=["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"])

print(df_cancer['target'].value_counts())
sns.countplot(data=df_cancer, x="target")

for i in range(0, len(num_cols), 10):
    plt.figure(figsize=(20, 10))
    for j in range(10):
        if i + j < len(num_cols):
            plt.subplot(2, 5, j + 1)
            col = num_cols[i + j]
            sns.histplot(df_cancer[col], kde=True, color='b', bins=30)
            plt.title(f'Histogram of {col}')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 10))
    for j in range(10):
        if i + j < len(num_cols):
            plt.subplot(2, 5, j + 1)
            col = num_cols[i + j]
            sns.boxplot(x=df_cancer[col], color='b')
            plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()

sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=df_cancer)

plt.figure(figsize=(20, 10))
sns.heatmap(df_cancer.corr(), annot=True)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
    if plot:
        sns.set(rc={"figure.figsize": (9, 9)})
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="magma")
        plt.show(block=True)
    return corr, print(f"Drop List: {drop_list}")


high_correlated_cols(df_cancer, plot=True)

## Checking Outlier


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df_cancer, col))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


for col in num_cols:
    replace_with_thresholds(df_cancer, col)


### Preprocessing and Model Training

X = df_cancer.drop(["target"], axis=1)
y = df_cancer["target"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=10)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict))


### Hyperparameter Tuning

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(),
                    param_grid,
                    refit=True,
                    verbose=4)

grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
cm2 = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm2, annot=True)
print(classification_report(y_test,grid_predictions))





