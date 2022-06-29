################################new section classification
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import nltk
from nltk.stem import PorterStemmer
porter = PorterStemmer()
from nltk.corpus import wordnet
nltk.download('omw-1.4')
import numpy as np
from sklearn.model_selection import train_test_split

import pickle

from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

import sys
measure = str(sys.argv[1])
print(measure)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--n", help="SDG K")
# args = parser.parse_args()
#
# number = int(args.n)



def stem_sentences(x):
    tokenized_words = x.split(" ")
    tokenized_sentence = []
    for word in tokenized_words:
        if len(wordnet.synsets(word)) != 0:
            tokenized_sentence.append(porter.stem(word))
    tokenized_sentence = " ".join(tokenized_sentence)
    return tokenized_sentence

# creating bag of words representations from description
# Create a Bag of Words Model with Sklearn
# import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
def get_BoW(df_wiki_node, column_name, param1=5, param2=.95):
    corpus = df_wiki_node[column_name]
    # sentence_1="*&^$This is a good job.{{I will not miss it for anything"
    # sentence_2="This is not good at all}}, hello my name misses a w"

#     CountVec = CountVectorizer(ngram_range=(1,2), # to use bigrams ngram_range=(2,2)
#                                stop_words='english')
    CountVec = CountVectorizer(min_df=param1,max_df=param2, ngram_range=(1, 2), stop_words='english')
    #transform
    Count_data = CountVec.fit_transform(corpus.values.tolist())

    #create dataframe
    BoW_dataframe=pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names_out())
    # print(BoW_dataframe)
    return BoW_dataframe


# number = 1
all_scores_all_SDGs = []
# list(set(range(1,18))-set([13]))
for number in range(1, 16):
    print("SDG ", number, " is calculating ...... ")
    msci = pd.read_csv("./data/msci.csv")
    msci2 = pd.read_csv("./data/msci2.csv")

    variable6 = "GICS Sector"

    if number >= 10:
        variable5 = "SDG_{}_PROD_ALIGNMENT".format(number)
    else:
        variable5 = "SDG_0{}_PROD_ALIGNMENT".format(number) # another thing

    SDG1 = msci[["Company Name", "Company ID"]].dropna()
    SDG2 = msci2[["ISSUER_NAME", "Figi", variable5]].dropna()

    df_label = SDG1.merge(SDG2, left_on="Company ID", right_on="Figi")[["Company Name", variable5]]
    df_label = df_label.rename(columns = {"Company Name": "company"})

    df_wiki = pd.read_csv("./temp_data/wiki/wiki_product_info.csv",sep="\t")
    df_wiki["product_info"] = df_wiki["product_info"].progress_apply(stem_sentences)
    df_merge = df_label.merge(df_wiki[["company","product_info"]],on="company").dropna()
    df_sector = pd.read_csv("./data/Fundamental.csv")[["Company Name",variable6]].rename(columns={"Company Name": "company"})
    df_merge2 = df_merge.merge(df_sector,on="company").dropna()
    df_entail = pd.read_csv("./temp_data/entail/entail_SDG_{}.csv".format(number),sep="\t")
    df_entail["report_evidence"] = df_entail.groupby("company")["statement"].transform(lambda x: ','.join(x))
    df_evidence = df_entail[["company","statement"]].drop_duplicates().rename(columns = {"statement":"report_evidence"})
    df_merge3 = df_merge2.merge(df_evidence,on="company",how="left")
    df_merge3["report_evidence"] = df_merge3["report_evidence"].fillna("nothing")
    df_merge3["stem_product_info"] = df_merge3["product_info"].progress_apply(stem_sentences)
    df_merge3["stem_report_evidence"] = df_merge3["report_evidence"].progress_apply(stem_sentences)


    labels = df_merge3[variable5].values
    features1 = pd.get_dummies(df_merge3[variable6])
    features2 = get_BoW(df_merge3, "stem_product_info", param1 = 10)
    features3 = get_BoW(df_merge3, "stem_report_evidence", param1 = 0)

    # if number == 13:
    #     features3 = get_BoW(df_merge3, "stem_report_evidence", param1 = 10, )
    # else:
    #     features3 = get_BoW(df_merge3, "stem_report_evidence", param1 = 5)
    print(features2.shape, features3.shape)
    if features2.shape[1] > 2000:
        features2 = get_BoW(df_merge3, "stem_product_info", param1 = 20)

    if features3.shape[1] > 2000:
        features3 = get_BoW(df_merge3, "stem_report_evidence", param1 = 10)

    print(features2.shape, features3.shape)
    all_scores = []
    ############################################# round 1
    print("round1")
    features = features1
    # features = features2
    # features = features3
    # features = np.concatenate((features1, features2),1)
    # features = np.concatenate((features1, features3),1)
    # features = np.concatenate((features2, features3),1)
    # features = np.concatenate((features1, features2, features3),1)

    X = features
    y = labels
    # define pipeline
    steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print('F1 Score: %.3f' % score)
    all_scores.append(score)

    ############################################# round 2
    print("round2")
    # features = features1
    features = features2
    # features = features3
    # features = np.concatenate((features1, features2),1)
    # features = np.concatenate((features1, features3),1)
    # features = np.concatenate((features2, features3),1)
    # features = np.concatenate((features1, features2, features3),1)

    X = features
    y = labels
    # define pipeline
    steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print('F1 Score: %.3f' % score)
    all_scores.append(score)

    ############################################# round 3
    print("round3")
    # features = features1
    # features = features2
    features = features3
    # features = np.concatenate((features1, features2),1)
    # features = np.concatenate((features1, features3),1)
    # features = np.concatenate((features2, features3),1)
    # features = np.concatenate((features1, features2, features3),1)

    X = features
    y = labels
    # define pipeline
    steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print('F1 Score: %.3f' % score)
    all_scores.append(score)

    ############################################# round 4
    print("round4")
    # features = features1
    # features = features2
    # features = features3
    features = np.concatenate((features1, features2),1)
    # features = np.concatenate((features1, features3),1)
    # features = np.concatenate((features2, features3),1)
    # features = np.concatenate((features1, features2, features3),1)

    X = features
    y = labels
    # define pipeline
    steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print('F1 Score: %.3f' % score)
    all_scores.append(score)

    ############################################# round 5
    print("round5")
    # features = features1
    # features = features2
    # features = features3
    # features = np.concatenate((features1, features2),1)
    features = np.concatenate((features1, features3),1)
    # features = np.concatenate((features2, features3),1)
    # features = np.concatenate((features1, features2, features3),1)

    X = features
    y = labels
    # define pipeline
    steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print('F1 Score: %.3f' % score)
    all_scores.append(score)

    ############################################# round 6
    print("round6")
    # features = features1
    # features = features2
    # features = features3
    # features = np.concatenate((features1, features2),1)
    # features = np.concatenate((features1, features3),1)
    features = np.concatenate((features2, features3),1)
    # features = np.concatenate((features1, features2, features3),1)

    X = features
    y = labels
    # define pipeline
    steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print('F1 Score: %.3f' % score)
    all_scores.append(score)

    ############################################# round 7
    print("round7")
    # features = features1
    # features = features2
    # features = features3
    # features = np.concatenate((features1, features2),1)
    # features = np.concatenate((features1, features3),1)
    # features = np.concatenate((features2, features3),1)
    features = np.concatenate((features1, features2, features3),1)

    X = features
    y = labels
    # define pipeline
    steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print('F1 Score: %.3f' % score)
    all_scores.append(score)


    all_scores_all_SDGs.append(all_scores)



with open('./results/product/{}.pkl'.format(measure),'wb') as f:
    pickle.dump(all_scores_all_SDGs, f, protocol=pickle.HIGHEST_PROTOCOL)

# import pickle
# with open("./data/embeddings_all1.pkl", "rb") as input_file:
#     e1 = pickle.load(input_file)
