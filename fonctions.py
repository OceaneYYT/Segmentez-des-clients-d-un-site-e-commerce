import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import kstest
from scipy.stats import normaltest
from IPython.display import display


''' Notebook nettoyage '''


def missing_cells(df):
    '''Calcule le nombre de cellules manquantes sur le data set total.
    Keyword arguments:
    df -- le dataframe

    return : le nombre de cellules manquantes de df
    '''
    return df.isna().sum().sum()


def missing_cells_perc(df):
    '''Calcule le pourcentage de cellules manquantes sur le data set total.
    Keyword arguments:
    df -- le dataframe

    return : le pourcentage de cellules manquantes de df
    '''
    return df.isna().sum().sum()/(df.size)


def missing_general(df):
    '''Donne un aperçu général du nombre de données manquantes dans le data frame.
    Keyword arguments:
    df -- le dataframe
    '''
    print('Nombre total de cellules manquantes :', missing_cells(df))
    print('Nombre de cellules manquantes en % : {:.2%}'
          .format(missing_cells_perc(df)))


def valeurs_manquantes(df):
    '''Prend un data frame en entrée et créer en sortie un dataframe contenant
    le nombre de valeurs manquantes et leur pourcentage pour chaque variables.
    Keyword arguments:
    df -- le dataframe

    return : dataframe contenant le nombre de valeurs manquantes et
    leur pourcentage pour chaque variable
    '''
    tab_missing = pd.DataFrame(columns=['Variable',
                                        'Missing values',
                                        'Missing (%)'])
    tab_missing['Variable'] = df.columns
    missing_val = list()
    missing_perc = list()

    for var in df.columns:
        nb_miss = missing_cells(df[var])
        missing_val.append(nb_miss)
        perc_miss = missing_cells_perc(df[var])
        missing_perc.append(perc_miss)

    tab_missing['Missing values'] = list(missing_val)
    tab_missing['Missing (%)'] = list(missing_perc)
    return tab_missing


def bar_missing(df):
    '''Affiche le barplot présentant le nombre de données présentes par variable.
    Keyword arguments:
    df -- le dataframe
    '''
    msno.bar(df)
    plt.title('Nombre de données présentes par variable', size=15)
    plt.show()


def barplot_missing(df):
    '''Affiche le barplot présentant le pourcentage de
    données manquantes par variable.
    Keyword arguments:
    df -- le dataframe
    '''
    proportion_nan = df.isna().sum()\
        .divide(df.shape[0]/100).sort_values(ascending=False)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 30))
    sns.barplot(y=proportion_nan.index, x=proportion_nan.values)
    plt.title('Pourcentage de données manquantes par variable', size=15)
    plt.show()


def drop_lignes(df, index):
    '''Supprime les lignes des index donnés en argument.
    Keyword arguments:
    df -- le dataframe
    index -- les index des lignes qu'on souhaite supprimer.
    '''
    df.drop(index, axis=0, inplace=True, errors='ignore')
    print('Suppression effectuée')


def df_agg_cust(df):
    '''Prend un data frame en entrée et retourne un dataframe agrégé et
    groupé en fonction de l'identifiant client unique.
    Keyword arguments:
    df -- le dataframe

    return : dataframe agrégé et groupé en fonction de
    l'identifiant client unique
    '''

    date_max = df['order_purchase_timestamp'].max()
    # date_min = df['order_purchase_timestamp'].min()

    df['mean_shipping_time'] = (df['order_delivered_customer_date']
                                - df['order_purchase_timestamp'])
    df['mean_delivery_delay'] = (df['order_delivered_customer_date']
                                 - df['order_estimated_delivery_date'])

    data_cust = df.groupby('customer_unique_id').agg(
        order_purchase_timestamp=('order_purchase_timestamp', np.max),
        order_delivered_customer_date=('order_delivered_customer_date',
                                       np.max),
        order_estimated_delivery_date=('order_estimated_delivery_date',
                                       np.max),
        time_since_first_order=('order_purchase_timestamp',
                                lambda x: (date_max - np.min(x)).days),
        time_since_last_order=('order_purchase_timestamp',
                               lambda x: (date_max - np.max(x)).days),
        mean_shipping_time=('mean_shipping_time',
                            lambda x: (np.mean(x).days)),
        mean_delivery_delay=('mean_delivery_delay',
                             lambda x: (np.mean(x).days)),
        customer_city=('customer_city', lambda x: x.mode()[0]),
        customer_state=('customer_state', lambda x: x.mode()[0]),
        nb_total_order=('order_id', 'count'),
        nb_total_item=('product_id', 'count'),
        total_price=('price', np.sum),
        mean_price=('price', np.mean),
        total_freight_value=('freight_value', np.sum),
        mean_freight_value=('freight_value', np.mean),
        payment_type=('payment_type', lambda x: x.mode()[0]),
        mean_payment_installments=('payment_installments', np.mean),
        total_payment_value=('payment_value', np.sum),
        mean_payment_value=('payment_value', np.mean),
        mean_review_score=('review_score', np.mean),
        seller_city=('seller_city', lambda x: x.mode()[0]),
        seller_state=('seller_state', lambda x: x.mode()[0]),
        cat=('product_category_name_english', lambda x: x.mode()[0])
    )

    return data_cust


def multi_boxplot(df, long, larg):
    ''' Affiche indépendamment tous les boxplots des variables sélectionnées.
    Keyword arguments:
    df -- le dataframe
    long -- nombre de figure en longueur
    larg -- nombre de figure en largeur
    '''
    fig, axs = plt.subplots(long, larg, figsize=(20, 40))
    axs = axs.ravel()

    for i, col in enumerate(df.columns):
        sns.boxplot(x=df[col], ax=axs[i])
    fig.suptitle('Boxplot pour chaque variable quantitative')
    plt.show()


def distribution(df, colonnes, long, larg):
    ''' Affiche les histogrammes pour chaque variable renseignée.
    Keyword arguments:
    df -- le dataframe
    colonnes -- variables à afficher
    long -- nombre de figure en longueur
    larg -- nombre de figure en largeur
    '''
    fig, axs = plt.subplots(long, larg, figsize=(20, 40))
    axs = axs.ravel()

    for i, col in enumerate(colonnes):
        sns.histplot(data=df, x=col, bins=30, kde=True, ax=axs[i])

    fig.suptitle('Distribution pour chaque variable quantitative')
    plt.show()


def test_normalite(df, colonnes, level):
    ''' Calcul les différents tests de normalité pour
    chacune des variables passées en paramètres
    et les affiche.
    Keyword arguments:
    df -- le dataframe
    colonnes -- variables pour lesquelles calculer les tests
    level -- niveau de confiance
    '''
    for col in colonnes:
        print("Tests de normalité pour la variable {}.".format(col))
        tests = [shapiro, anderson, normaltest, kstest]
        index = ['Shapiro Wilk', 'Anderson-Darling',
                 "K2 de D'Agostino", 'Kolmogorov-Smirnov']
        tab_result = pd.DataFrame(columns=['Stat', 'p-value', 'Resultat'],
                                  index=index)

        for i, fc in enumerate(tests):
            if fc == anderson:
                result = fc(df[col])
                tab_result.loc[index[i], 'Stat'] = result.statistic
                if result.statistic < result.critical_values[2]:
                    tab_result.loc[index[i], 'Resultat'] = 'H0'
                if result.statistic > result.critical_values[2]:
                    tab_result.loc[index[i], 'Resultat'] = 'H1'

            elif fc == kstest:
                stat, p = fc(df[col], cdf='norm')
                tab_result.loc[index[i], 'Stat'] = stat
                tab_result.loc[index[i], 'p-value'] = p
                if p < level:
                    tab_result.loc[index[i], 'Resultat'] = 'H1'
                if p > level:
                    tab_result.loc[index[i], 'Resultat'] = 'H0'

            else:
                stat, p = fc(df[col])
                tab_result.loc[index[i], 'Stat'] = stat
                tab_result.loc[index[i], 'p-value'] = p
                if p < level:
                    tab_result.loc[index[i], 'Resultat'] = 'H1'
                if p > level:
                    tab_result.loc[index[i], 'Resultat'] = 'H0'

        print(tab_result)
        print("-"*70)


def bar_plot(df, colonnes, long, larg):
    ''' Affiche les bar plots pour chaque variable renseignée.
    Keyword arguments:
    df -- le dataframe
    colonnes -- variables à afficher
    long -- nombre de figure en longueur
    larg -- nombre de figure en largeur
    '''
    fig = plt.figure(figsize=(40, 40))
    for i, col in enumerate(colonnes, 1):
        ax = fig.add_subplot(long, larg, i)
        count = df[col].value_counts()
        count.plot(kind="bar", ax=ax)
        plt.xticks(rotation=90, ha='right', fontsize=20)
        ax.set_title(col, fontsize=20)
    plt.tight_layout(pad=2)
    plt.show()


def pie_plot(df, colonnes):
    '''Affiche un pie plot présentant la répartition de la variable renseignée.
    Keyword arguments:
    df -- le dataframe
    colonnes -- variables à afficher
    '''
    for col in colonnes:
        labels = list(df[col].value_counts().sort_index().index.astype(str))
        count = df[col].value_counts().sort_index()

        plt.figure(figsize=(10, 10))
        plt.pie(count, autopct='%1.2f%%')
        plt.title('Répartition de {}'.format(col), size=20)
        plt.legend(labels)
        plt.show()


def heat_map(df_corr):
    '''Affiche la heatmap.
    Keyword arguments:
    df_corr -- le dataframe des corrélations
    '''
    plt.figure(figsize=(15, 10))
    sns.heatmap(df_corr, annot=True, linewidth=.5)
    plt.title("Heatmap")


def tests_corr(df, colonnes, var_comparaison):
    ''' Calcul les différents tests de corrélation pour chacun des couples
    de variables passés en paramètres.
    Keyword arguments:
    df -- le dataframe
    colonnes -- variables à afficher
    var_comparaison -- variable avec laquelle comparer
    '''
    for col in colonnes:
        print("Tests de corrélation pour la variable {}"
              "par rapport à la variable {}.".format(col, var_comparaison))
        tests = [pearsonr, spearmanr, kendalltau]
        index = ['Pearson', 'Spearman', 'Kendall']
        tab_result = pd.DataFrame(columns=['Stat', 'p-value'], index=index)

        for i, fc in enumerate(tests):
            stat, p = fc(df[col], df[var_comparaison])
            tab_result.loc[index[i], 'Stat'] = stat
            tab_result.loc[index[i], 'p-value'] = p
        display(tab_result)
        print("-"*100)


def boxplot_relation(df, colonnes, var_comparaison, longueur,
                     largeur, ordre=None, outliers=True, option=False):
    '''Affiche les boxplot des colonnes en fonctions de var_comparaison.
    Keyword arguments:
    df -- le dataframe
    colonnes -- variables à afficher
    var_comparaison -- variable avec laquelle comparer
    longueur -- nombre de figure en longueur
    largeur -- nombre de figure en largeur
    ordre -- ordre dans lequel placer les valeurs catégorielles (default None)
    outliers -- afficher ou non les valeurs jugées
    comme outliers par le boxplot (default True)
    option -- afficher les labels de x avec une rotation de 90° (default False)
    '''
    fig = plt.figure(figsize=(40, 60))
    for i, col in enumerate(colonnes, 1):
        ax = fig.add_subplot(longueur, largeur, i)
        sns.boxplot(x=df[var_comparaison], y=df[col],
                    ax=ax, order=ordre, showfliers=outliers)
        if option:
            plt.xticks(rotation=90, ha='right')
    fig.suptitle('Boxplot de chaque target en fonction de {}'
                 .format(var_comparaison))
    plt.tight_layout(pad=4)
    plt.show()
