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
import numpy as np
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

### find connected components in a graph
# Python program to print connected
# components in an undirected graph
import sys
sys.setrecursionlimit(100000)

class Graph:

    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]

    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:

                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp

    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc



def get_category(x):
    if x == "Strongly Misaligned":
        return "Misaligned"
    elif x == "Strongly Aligned":
        return "Aligned"
    else:
        return x

def convert_report_evidence(x):
    if x == "nothing":
        return False
    else:
        return True

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
def get_BoW(df_wiki_node, column_name, param1=5, param2=.99):
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


import wordninja
def split_words(x):
    return " ".join(wordninja.split(x))

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    # nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == "JJ" or pos == "JJR" or pos == "JJS")]
    # nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == "JJ" or pos == "JJR" or pos == "JJS" or pos[0] == "V")]
    try:
        nouns = [word for word,pos in tags if (pos[0] == "N" or pos[0] == "J" or pos[1] == "V")]
        return " ".join(nouns).lower()
    except:
        return "nothing"

def get_score(prod, report, news):
    score = prod
    if report == "nothing":
        score += 0
    else:
        score += 1

    if news > 0:
        score += 1
    else:
        score -= 1
    return score

def adjust_score(score,graph):
    if graph < score and np.sign(graph) != np.sign(score):
        return score - 0.5
    elif graph > score and np.sign(graph) != np.sign(score):
        return score + 0.5
    else:
        return score

df_merge = pd.read_csv("./results_news/features4.csv")
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
    # number = 2
    print("SDG ", number, " is calculating ...... ")
    msci = pd.read_csv("./data/msci.csv")
    msci2 = pd.read_csv("./data/msci2.csv").rename(columns={"SDG_03_OPS_ALIGNMENT":"SDG_03_OPER_ALIGNMENT"})

    variable6 = "GICS Industry"

    if number >= 10:
        variable5 = "SDG_{}_PROD_ALIGNMENT".format(number)
    else:
        variable5 = "SDG_0{}_PROD_ALIGNMENT".format(number) # another thing

    SDG1 = msci[["Company Name", "Company ID"]].dropna()
    SDG2 = msci2[["ISSUER_NAME", "Figi", variable5]].dropna()

    df_label = SDG1.merge(SDG2, left_on="Company ID", right_on="Figi")[["Company Name", variable5]]
    df_label = df_label.rename(columns = {"Company Name": "company"})

    df_sector = pd.read_csv("./data/Fundamental.csv")[["Company Name",variable6]].rename(columns={"Company Name": "company"})
    df_merge2 = df_merge.merge(df_sector,on="company", how="right")
    df_merge3 = df_merge2.merge(df_label, on="company", how="right")
    # df_merge3["concat_header_cleaned"] = df_merge3["concat_header_cleaned"].fillna("nothing")
    # df_merge3["concat_header_cleaned"] = df_merge3["concat_header_cleaned"].progress_apply(split_words)
    # df_merge3["concat_header_cleaned"] = df_merge3["concat_header_cleaned"].progress_apply(clean_text)

    df_merge3 = df_merge3.dropna(subset=[variable6,variable5])
    df_merge3 = df_merge3.fillna(df_merge3.groupby(variable6).transform("mean"))
    # added
    df_wiki = pd.read_csv("./temp_data/wiki/wiki_product_info2.csv",sep="\t")
    df_merge3 = df_merge3.merge(df_wiki[["company","product_info"]],on="company").dropna()
    # added
    df_entail = pd.read_csv("./temp_data/entail/entail_SDG_{}.csv".format(number),sep="\t")
    df_entail["report_evidence"] = df_entail.groupby("company")["statement"].transform(lambda x: ','.join(x))
    df_evidence = df_entail[["company","statement"]].drop_duplicates().rename(columns = {"statement":"report_evidence"})
    df_merge3 = df_merge3.merge(df_evidence,on="company",how="left")
    df_merge3["report_evidence"] = df_merge3["report_evidence"].fillna("nothing")
    df_merge3["report_evidence"] = df_merge3["report_evidence"].progress_apply(split_words)
    # df_merge3["report_evidence"] = df_merge3["report_evidence"].progress_apply(clean_text)

    # df_merge3["stem_product_info"] = df_merge3["product_info"].progress_apply(stem_sentences)
    # df_merge3["stem_report_evidence"] = df_merge3["report_evidence"].progress_apply(stem_sentences)
    encoded_dict = {"Strongly Misaligned":-2,'Misaligned':1,"Neutral":0,"Aligned":1,"Strongly Aligned":2}
    df_merge3[variable5] = df_merge3[variable5].map(encoded_dict)
    # df_merge3 = df_merge3[df_merge3[variable5]!=2]
    # df_merge3 = df_merge3[df_merge3[variable6]=="Banks"]

    labels = df_merge3[variable5]
    features1 = pd.get_dummies(df_merge3[variable6])
    # features2 = get_BoW(df_merge3, "concat_header_cleaned", param1 = 10, param2 = 40)
    # features3 = df_merge3[column_features]
    features4 = get_BoW(df_merge3, "product_info", param1 = 10, param2 = 40)
    # if number == 13:
    #     features5 = get_BoW(df_merge3, "report_evidence", param1 = 10)
    # else:
    #     features5 = get_BoW(df_merge3, "report_evidence", param1 = 0)

    # all_scores = []

    df_wiki_id = pd.read_csv("./wiki_data/wikidata.csv")
    df_merge3 = df_merge3.merge(df_wiki_id[["company","wikidata_id"]], on="company", how="left").dropna()


    df_wiki_node = df_merge3[["company", variable5,"report_evidence", "overall_score_mean", "wikidata_id"]]
    df_wiki_node["new_id"] = range(len(df_wiki_node))

    # df_node = pd.read_csv("GCN/data/wiki_graph_data_2hop_description.csv")
    df_graph = pd.read_csv("./wiki_data/wiki_graph_data_2hop.csv")
    df_graph = df_graph[(df_graph.wikidata_id_start.isin(df_merge3.wikidata_id.values))&(df_graph.wikidata_id_end.isin(df_merge3.wikidata_id.values))]

    keys_list = df_wiki_node["wikidata_id"].values.tolist()
    values_list = df_wiki_node["new_id"].values.tolist()
    zip_iterator = zip(keys_list, values_list)
    a_dictionary = dict(zip_iterator)
    map_dictionary = {**a_dictionary}

    df_graph["start_new_id"] = df_graph["wikidata_id_start"].map(map_dictionary )
    df_graph["end_new_id"] = df_graph["wikidata_id_end"].map(map_dictionary )

    df_wiki_node["score"] = df_wiki_node.progress_apply(lambda row : get_score(row[variable5],row["report_evidence"], row['overall_score_mean']), axis = 1)

    df_graph["start_new_id"] = df_graph["start_new_id"].astype(int)
    df_graph["end_new_id"] = df_graph["end_new_id"].astype(int)

    g = Graph(len(df_wiki_node))
    for x in df_graph[["start_new_id", "end_new_id"]].values:
        g.addEdge(x[0], x[1])
    cc = g.connectedComponents()


    arr_list = []
    for ele in cc:
        if len(ele) > 1:
            arr = []
            for ele2 in ele:
                arr.append(df_wiki_node[df_wiki_node.new_id == ele2].score.values[0])
            mean_score = np.mean(arr)
            for ele2 in ele:
                arr_list.append([ele2, mean_score])


    d = {}

    for k, v in arr_list:
        d[k] = v
    df_wiki_node["graph"] = df_wiki_node["new_id"].map(d)
    df_wiki_node["graph"] = df_wiki_node.graph.fillna(df_wiki_node.score)
    df_wiki_node["final_score"] = df_wiki_node.progress_apply(lambda row : adjust_score(row["score"],row["graph"]), axis = 1)
    df_wiki_node["report_evidence_binary"] = df_wiki_node["report_evidence"].progress_apply(convert_report_evidence)
    df_wiki_node[["company","wikidata_id", variable5, "report_evidence_binary", "report_evidence", "overall_score_mean", "score", "graph", "final_score"]].rename(columns={"overall_score_mean":"average_sentiment_impactful", "graph":"graph_group_average", variable5: "product_score", "report_evidence":"report_evidence_content", "score":"score_before_graph"}).to_csv("./generate/SDG_{}_generated_score.csv".format(number))
