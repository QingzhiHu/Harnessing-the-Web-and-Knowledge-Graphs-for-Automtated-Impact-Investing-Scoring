################################new section classification
import warnings
warnings.filterwarnings("ignore")
import sys
measure = str(sys.argv[1])
print(measure)
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

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--n", help="SDG K")
# args = parser.parse_args()
#
# number = int(args.n)
def get_category(x):
    if x == "Strongly Misaligned":
        return "Misaligned"
    elif x == "Strongly Aligned":
        return "Aligned"
    else:
        return x


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

df_merge = pd.read_csv("./temp_data2/news_features.csv")
column_features = ['magnitude_sum', 'magnitude_mean', 'magnitude_std',
       'magnitude_median', 'magnitude_var', 'magnitude_amin', 'magnitude_amax',
       'magnitude_percentile_5', 'magnitude_percentile_95',
       'magnitude_percentile_10', 'magnitude_percentile_90', 'score_sum',
       'score_mean', 'score_std', 'score_median', 'score_var', 'score_amin',
       'score_amax', 'score_percentile_5', 'score_percentile_95',
       'score_percentile_10', 'score_percentile_90', 'numMentions_sum',
       'numMentions_mean', 'numMentions_std', 'numMentions_median',
       'numMentions_var', 'numMentions_amin', 'numMentions_amax',
       'numMentions_percentile_5', 'numMentions_percentile_95',
       'numMentions_percentile_10', 'numMentions_percentile_90',
       'avgSalience_sum', 'avgSalience_mean', 'avgSalience_std',
       'avgSalience_median', 'avgSalience_var', 'avgSalience_amin',
       'avgSalience_amax', 'avgSalience_percentile_5',
       'avgSalience_percentile_95', 'avgSalience_percentile_10',
       'avgSalience_percentile_90', 'overall_score_sum', 'overall_score_mean',
       'overall_score_std', 'overall_score_median', 'overall_score_var',
       'overall_score_amin', 'overall_score_amax',
       'overall_score_percentile_5', 'overall_score_percentile_95',
       'overall_score_percentile_10', 'overall_score_percentile_90']
# number = 1
all_scores_all_SDGs = []
for number in range(1,18):
    print("SDG ", number, " is calculating ...... ")
    msci = pd.read_csv("./data/msci.csv")
    msci2 = pd.read_csv("./data/msci2.csv").rename(columns={"SDG_03_OPS_ALIGNMENT":"SDG_03_OPER_ALIGNMENT"})

    variable6 = "GICS Sector"

    if number >= 10:
        variable5 = "SDG_{}_NET_ALIGNMENT".format(number)
    else:
        variable5 = "SDG_0{}_NET_ALIGNMENT".format(number) # another thing

    SDG1 = msci[["Company Name", "Company ID"]].dropna()
    SDG2 = msci2[["ISSUER_NAME", "Figi", variable5]].dropna()

    df_label = SDG1.merge(SDG2, left_on="Company ID", right_on="Figi")[["Company Name", variable5]]
    df_label = df_label.rename(columns = {"Company Name": "company"})

    df_sector = pd.read_csv("./data/Fundamental.csv")[["Company Name",variable6]].rename(columns={"Company Name": "company"})
    df_merge2 = df_merge.merge(df_sector,on="company", how="right")
    df_merge3 = df_merge2.merge(df_label, on="company", how="right")
    df_merge3["concat_header_cleaned"] = df_merge3["concat_header_cleaned"].fillna("nothing")
    df_merge3 = df_merge3.dropna(subset=[variable6,variable5])
    df_merge3 = df_merge3.fillna(df_merge3.groupby(variable6).transform("mean"))
    # added
    df_wiki = pd.read_csv("./temp_data/wiki/wiki_product_info.csv",sep="\t")
    df_wiki["product_info"] = df_wiki["product_info"].progress_apply(stem_sentences)
    df_merge3 = df_merge3.merge(df_wiki[["company","product_info"]],on="company").dropna()
    # added
    df_entail = pd.read_csv("./temp_data/entail/entail_SDG_{}.csv".format(number),sep="\t")
    df_entail["report_evidence"] = df_entail.groupby("company")["statement"].transform(lambda x: ','.join(x))
    df_evidence = df_entail[["company","statement"]].drop_duplicates().rename(columns = {"statement":"report_evidence"})
    df_merge3 = df_merge3.merge(df_evidence,on="company",how="left")
    df_merge3["report_evidence"] = df_merge3["report_evidence"].fillna("nothing")
    df_merge3["stem_product_info"] = df_merge3["product_info"].progress_apply(stem_sentences)
    df_merge3["stem_report_evidence"] = df_merge3["report_evidence"].progress_apply(stem_sentences)


    labels = df_merge3[variable5].values
    features1 = pd.get_dummies(df_merge3[variable6])
    features2 = get_BoW(df_merge3, "concat_header_cleaned", param1 = 10)
    features3 = df_merge3[column_features]
    features4 = get_BoW(df_merge3, "stem_product_info", param1 = 10)
    # if number == 13:
    #     features5 = get_BoW(df_merge3, "stem_report_evidence", 10)
    # else:
    #     features5 = get_BoW(df_merge3, "stem_report_evidence", param1 = 0)

    if features2.shape[1] > 2000:
        features2 = get_BoW(df_merge3, "concat_header_cleaned", param1 = 30)
    if features4.shape[1] > 2000:
        features4 = get_BoW(df_merge3, "stem_product_info", param1 = 30)
    # if features5.shape[1] > 2000:
    #     features5 = get_BoW(df_merge3, "stem_report_evidence", 20)

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

    ############################################# round 1
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

    ############################################# round 1
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

    ############################################# round 1
    print("round4")
    # features = features1
    # features = features2
    # features = features3
    features = features4
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

    # ############################################# round 1
    # print("round5")
    # # features = features1
    # # features = features2
    # # features = features3
    # features = features5
    # # features = np.concatenate((features1, features2),1)
    # # features = np.concatenate((features1, features3),1)
    # # features = np.concatenate((features2, features3),1)
    # # features = np.concatenate((features1, features2, features3),1)
    #
    # X = features
    # y = labels
    # # define pipeline
    # steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    # pipeline = Pipeline(steps=steps)
    # # evaluate pipeline
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    # score = mean(scores)
    # print('F1 Score: %.3f' % score)
    # all_scores.append(score)

    # ############################################# round 2
    # print("round6")
    # # features = features1
    # # features = features2
    # # features = features3
    # features = np.concatenate((features1, features2, features4),1)
    # # features = np.concatenate((features1, features3),1)
    # # features = np.concatenate((features2, features3),1)
    # # features = np.concatenate((features1, features2, features3),1)
    #
    # X = features
    # y = labels
    # # define pipeline
    # steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    # pipeline = Pipeline(steps=steps)
    # # evaluate pipeline
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    # score = mean(scores)
    # print('F1 Score: %.3f' % score)
    # all_scores.append(score)
    #
    # ############################################# round 3
    # print("round3")
    # # features = features1
    # # features = features2
    # # features = features3
    # features = np.concatenate((features1, features3, features4),1)
    # # features = np.concatenate((features1, features3),1)
    # # features = np.concatenate((features2, features3),1)
    # # features = np.concatenate((features1, features2, features3),1)
    #
    # X = features
    # y = labels
    # # define pipeline
    # steps = [('over', RandomOverSampler()), ('model', BalancedRandomForestClassifier())]
    # pipeline = Pipeline(steps=steps)
    # # evaluate pipeline
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    # score = mean(scores)
    # print('F1 Score: %.3f' % score)
    # all_scores.append(score)

    ############################################# round 4
    print("round6")
    # features = features1
    # features = features2
    # features = features3
    features = np.concatenate((features1, features2, features3, features4),1)
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


    all_scores_all_SDGs.append(all_scores)



with open('./results/all/{}.pkl'.format(measure),'wb') as f:
    pickle.dump(all_scores_all_SDGs, f, protocol=pickle.HIGHEST_PROTOCOL)

# import pickle
# with open("./data/embeddings_all1.pkl", "rb") as input_file:
#     e1 = pickle.load(input_file)
