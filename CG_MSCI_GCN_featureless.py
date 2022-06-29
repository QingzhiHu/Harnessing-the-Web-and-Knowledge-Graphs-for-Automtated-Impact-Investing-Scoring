import pandas as pd
df_descrp = pd.read_csv("./wiki_data/wiki_graph_data_2hop_description.csv")

import sys
frac_data = float(sys.argv[1])
measure = str(sys.argv[2])
# print(measure)

# run_time = int(sys.argv[2])
################################new section classification
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.data import Data
import torch
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import nltk
from nltk.stem import PorterStemmer
porter = PorterStemmer()
from nltk.corpus import wordnet
nltk.download('omw-1.4')
import numpy as np
# from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
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
def get_category(x):
    if x == "Strongly Misaligned":
        return "Misaligned"
    elif x == "Strongly Aligned":
        return "Aligned"
    else:
        return x

import string
import re
import nltk
nltk.download('wordnet')
nltk.download('punkt')
def extract_statements(line):
    try:
        line = line.replace("|", " ")
        line = line.replace("=", " ")
        line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
        # removing trailing spaces
        line = line.strip()
        # words may be split between lines, ensure we link them back together
        line = re.sub(r'\s?-\s?', '-', line)
        # remove space prior to punctuation
        line = re.sub(r'\s?([,:;\.])', r'\1', line)
        # ESG contains a lot of figures that are not relevant to grammatical structure
        line = re.sub(r'\d{5,}', r' ', line)
        # remove mentions of URLs
        line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
        # remove multiple spaces
        line = re.sub(r'\s+', ' ', line)
        # remove multiple dot
        line = re.sub(r'\.+', '.', line)
        sentences = []
        # split paragraphs into well defined sentences using nltk
        for part in nltk.sent_tokenize(line):
            sentences.append(str(part).strip())

        my_string = " ".join(sentences)
        my_string = my_string.replace("_", " ")
        new_string = my_string.translate(str.maketrans('', '', string.punctuation))
        return new_string
    except:
        return None


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
    CountVec = CountVectorizer(min_df=param1,max_df=param2, ngram_range=(1, 1), stop_words='english')
    #transform
    Count_data = CountVec.fit_transform(corpus.values.tolist())

    #create dataframe
    BoW_dataframe=pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
    # print(BoW_dataframe)
    return BoW_dataframe

import re
def find_category(x):
    try:
        return " ".join(re.findall(r'Category:([^\[\]]*)', x))
    except:
        return None

def convert_label_numeric(x):
    if x == "Strongly Misaligned":
        return 1
    if x == "Misaligned":
        return 2
    if x == "Neutral":
        return 3
    if x == "Aligned":
        return 4
    if x == "Strongly Aligned":
        return 5

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.x.shape[1], 16)
        self.conv2 = GCNConv(16, len(data.y.unique()))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


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



# import sys
# print(int(sys.argv[1]))
# number = int(sys.argv[1])
overall_all_scores = []
for i in range(1):
    print("run time ", i)
    all_scores = []
    for number in range(1,18):
        # try:
        print("calculating SDG: ", number)
        msci = pd.read_csv("./data/msci.csv")
        msci2 = pd.read_csv("./data/msci2.csv").rename(columns={"SDG_03_OPS_ALIGNMENT":"SDG_03_OPER_ALIGNMENT"})

        variable6 = "GICS Industry"

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
        df_wiki_id = pd.read_csv("./wiki_data/wikidata.csv")
        df_merge3 = df_merge3.merge(df_wiki_id[["company","wikidata_id"]], on="company", how="left").dropna()

        df_graph = pd.read_csv("./wiki_data/wiki_graph_data_2hop_cleaned.csv")
        # df_graph = df_graph[(df_graph.wikidata_id_start.isin(df_merge3.wikidata_id.values))|(df_graph.wikidata_id_end.isin(df_merge3.wikidata_id.values))]

        df_wiki_node = pd.DataFrame()
        df_wiki_node["wikidata_id"] = list(set(df_graph.wikidata_id_start.values.tolist()+df_graph.wikidata_id_end.values.tolist()))

        df_wiki_node = df_wiki_node.merge(df_merge3,on="wikidata_id",how="left").drop_duplicates(subset=["wikidata_id"])
        df_wiki_node["new_id"] = range(len(df_wiki_node))
        df_wiki_node = df_wiki_node.rename(columns={"wikidata_id":"wiki_id"})

        df_wiki_node["product_info"] = df_wiki_node["product_info"].progress_apply(extract_statements)
        df_wiki_node["product_info"] = df_wiki_node["product_info"].progress_apply(extract_statements)
        df_wiki_node = df_wiki_node.merge(df_descrp[["wiki_id","descriptions"]], on="wiki_id", how="left")

        df_wiki_node["descriptions_clean"] = df_wiki_node["descriptions"].progress_apply(find_category)
        df_wiki_node['descriptions_clean'] = df_wiki_node['descriptions_clean'].fillna(df_wiki_node['product_info'])
        df_wiki_node["descriptions_clean"] = df_wiki_node["descriptions_clean"].fillna("nothing")

        # df_wiki_node[variable5] = df_wiki_node[variable5].apply(convert_label_numeric)
        keys_list = df_wiki_node["wiki_id"].values.tolist()
        values_list = df_wiki_node["new_id"].values.tolist()
        zip_iterator = zip(keys_list, values_list)
        a_dictionary = dict(zip_iterator)
        map_dictionary = {**a_dictionary}

        df_graph["start_new_id"] = df_graph["wikidata_id_start"].map(map_dictionary )
        df_graph["end_new_id"] = df_graph["wikidata_id_end"].map(map_dictionary )

        # model
        data = Data()
        a = torch.tensor(df_graph[["end_new_id", "start_new_id"]].values, dtype=torch.long).t()
        b = torch.tensor(df_graph[["start_new_id", "end_new_id"]].values, dtype=torch.long).t()
        data.edge_index = torch.cat((a,b), 1)

        df_index = df_wiki_node[["new_id",variable5]].dropna()
        # shuffle
        result = df_index.sample(frac=1.0)
        # get the first two by group
        result = result.groupby(variable5).sample(frac=frac_data)
        result = result.sort_values(variable5)

        train_list = np.full(len(df_wiki_node), False)
        train_list[result.new_id.values] = True
        test_list = np.full(len(df_wiki_node), False)
        test_list[list(set(df_index.new_id.values) - set(result.new_id.values))] = True

        data.train_mask = torch.tensor(train_list)
        data.test_mask = torch.tensor(test_list)

        df_wiki_node[variable5] = df_wiki_node[variable5].fillna("nothing")
        df_wiki_node[variable5] = df_wiki_node[variable5].factorize()[0]
        data.y = torch.tensor(df_wiki_node[variable5].values, dtype=torch.int64)

        df_wiki_node[variable6] = df_wiki_node[variable6].fillna("another")

        del msci,msci2,SDG1,SDG2,df_label,df_sector,df_merge2,df_merge3,df_wiki,df_entail,df_wiki_id,df_graph

        df_wiki_node["concat_header_cleaned"] = df_wiki_node["concat_header_cleaned"].fillna("nothing")
        df_wiki_node[column_features] = df_wiki_node[column_features].fillna(0)
        df_wiki_node["stem_report_evidence"] = df_wiki_node["stem_report_evidence"].fillna("nothing")

        features1 = pd.get_dummies(df_wiki_node[variable6])
        features2 = get_BoW(df_wiki_node, "concat_header_cleaned", 20,60)
        features3 = df_wiki_node[column_features]
        features4 = get_BoW(df_wiki_node, "descriptions_clean", 20,60)
        # if number == 13:
        #     features5 = get_BoW(df_wiki_node, "stem_report_evidence", param1 = 20)
        # else:
        #     features5 = get_BoW(df_wiki_node, "stem_report_evidence", param1 = 0)

        features = np.concatenate((features1, features2, features3, features4), 1)
        # features = np.concatenate((features1, features4), 1)

        # data.x = torch.tensor(features, dtype=torch.float)
        data.x = torch.eye(df_wiki_node.shape[0])

        class GCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(data.x.shape[1], 16)
                self.conv2 = GCNConv(16, len(data.y.unique()))

            def forward(self, data):
                x, edge_index = data.x, data.edge_index

                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)

                return F.log_softmax(x, dim=1)

        # device = torch.device('cuda' if quit() else 'cpu')
        device = "cpu"
        print("device", device)
        model = GCN().to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
        losses = []
        model.train()

        best_score_train = []
        best_score_validation = []
        best_score_test = []
        for epoch in range(5000):
            if epoch % 1000 == 0:
                print(epoch)
            if epoch % 30 == 0:
                model.eval()
                pred = model(data).argmax(dim=1)

                acc_train = f1_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu(), average=measure)
                acc_test = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average=measure)

                # print(f'Accuracy: {acc_train:.4f}, {acc_test:.4f}')
                best_score_train.append(acc_train)
                best_score_test.append(acc_test)

                if acc_test < best_score_test[-1]:
                    break


            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()


        # something else comparison #########################################
        X = data.x.cpu().numpy()
        y = data.y.cpu().numpy()

        X_train = X[data.train_mask.cpu()]
        y_train = y[data.train_mask.cpu()]

        X_test = X[data.test_mask.cpu()]
        y_test = y[data.test_mask.cpu()]

        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        # X_test, y_test = ros.fit_resample(X_test, y_test)

        model = BalancedRandomForestClassifier()
        model.fit(X_train, y_train)
        expected_y  = y_test
        predicted_y = model.predict(X_test)

        comparison_score = f1_score(expected_y,predicted_y, average=measure)
        #####
        X_train = features1.values[data.train_mask.cpu()]
        y_train = y[data.train_mask.cpu()]
        X_test = features1.values[data.test_mask.cpu()]
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        expected_y  = y_test
        predicted_y = model.predict(X_test)

        comparison_score2 = f1_score(expected_y,predicted_y, average=measure)
        ###################################################################
        # save scores
        all_scores.append([number, max(best_score_train), max(best_score_test), comparison_score, comparison_score2, best_score_train[-1], best_score_test[-1]])
        print("best scores train & test", max(best_score_train), max(best_score_test), comparison_score, comparison_score2)
        print("score end train & test", best_score_train[-1], best_score_test[-1])
        # from matplotlib import pyplot as plt
        # fig = plt.figure()
        # plt.plot(best_score_train)
        # plt.plot(best_score_test)
        # # plt.show()
        # fig.savefig('./figures/SDG_{}.png'.format(number), dpi=fig.dpi)
        # except:
        #     all_scores.append(None)
    overall_all_scores.append(all_scores)


with open('./results/KG/msci_gcn_{}_{}_featuresless.pkl'.format(frac_data, measure),'wb') as f:
    pickle.dump(overall_all_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
