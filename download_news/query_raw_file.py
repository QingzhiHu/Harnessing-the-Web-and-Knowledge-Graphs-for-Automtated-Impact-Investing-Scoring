#!/usr/bin/env python
# coding: utf-8

# In[45]:


# https://blog.gdeltproject.org/announcing-the-global-entity-graph-geg-and-a-new-11-billion-entity-dataset/
import os
import sqlite3 as db
import pandas as pd
import json
from urllib.parse import urlparse
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="starting index", default=0)
parser.add_argument("--end", help="ending index", default=len(os.listdir("./data")))
args = parser.parse_args()

start = int(args.start)
end = int(args.end)

print("start: ", start, " end: ", end)

def parse_url(url):
    o = urlparse(url)
    return o.path

wiki = pd.read_csv("./new/companies_wikipedia_corrected.csv")
wiki["url_path"] = wiki.url.apply(parse_url)

df_list = []
for filename in tqdm(os.listdir(r"D:/raw/data/")[start:end]):
    try:
        data = []
        with open(r"D:/raw/data/"+filename, encoding="utf8") as f:
            for line in f:
                line = line.replace('''"wikipediaUrl": ,''', '''"wikipediaUrl": "None",''')
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        df1 = pd.concat([pd.DataFrame(x) for x in df['entities']], keys=df["url"]).reset_index(level=1, drop=True).reset_index()
        df1 = df1[(df1.type=="ORGANIZATION")&(~df1.wikipediaUrl.isnull())&(df1.wikipediaUrl!="None")]
        df1["url_path"] = df1.wikipediaUrl.apply(parse_url)
        df1 = df1[df1.url_path.isin(wiki["url_path"].values)]
        df1 = pd.merge(df[['date', 'url', 'lang', 'polarity', 'magnitude', 'score']], df1)
        df_merged = pd.merge(df1, wiki, on="url_path")[["company","date","url_x","lang","polarity","magnitude","score","numMentions","avgSalience","name","mid","wikipediaUrl"]].rename(columns={"url_x": "news_url", "name":"wiki_name"})
        # df_merged.to_csv("./data/"+".".join(filename.split(".")[:-1])+".csv")
        df_list.append(df_merged)
    except:
        print("this one doesn't have wiki url", filename)

# for filename in tqdm(os.listdir(r"D:/raw/data/")[start:end]):
#     try:
#         data = []
#         with open(r"D:/raw/data/"+filename, encoding="utf8") as f:
#             for line in f:
#                 line = line.replace('''"wikipediaUrl": ,''', '''"wikipediaUrl": "None",''')
#                 data.append(json.loads(line))
#         df = pd.DataFrame(data)
#         df_merged = pd.merge(df[['date', 'url', 'lang', 'polarity', 'magnitude', 'score']],pd.concat([pd.DataFrame(x) for x in df['entities']], keys=df["url"]).reset_index(level=1, drop=True).reset_index())
#         df1 = df_merged[(df_merged.type == "ORGANIZATION")&(~df_merged['wikipediaUrl'].isnull())]
#         df1["url_path"] = df1.wikipediaUrl.apply(parse_url)
#
#         df_merge2 = pd.merge(df1, wiki, on="url_path")
#         df_merge2 = df_merge2[["company","date","url_x","lang","polarity","magnitude","score","numMentions","avgSalience","name","mid","wikipediaUrl"]].rename(columns={"url_x": "news_url", "name":"wiki_name"})
#         df_list.append(df_merge2)
#     except:
#         print("this one doesn't have wiki url", filename)
#


final = pd.concat(df_list)
final.to_csv("./data/merged_filtered_news_{}_{}.csv".format(start, end), index=False)




# In[ ]:
