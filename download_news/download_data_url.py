#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="starting index")
parser.add_argument("--end", help="ending index")
args = parser.parse_args()

start = int(args.start)
end = int(args.end)


# In[67]:


import pandas as pd
news_data = pd.read_csv("./new/news_data.csv")
# news_data = news_data[["company", "date", "news_url","avgSalience"]].groupby("news_url").head(1)
news_data = news_data[news_data.avgSalience>=0.01]
news_data = news_data.iloc[start:end, :]
print("total number of urls is ", len(news_data))


# In[74]:


# news_data


# In[73]:


# start = 0
# end = len(news_data.news_url.values)
# end


# In[69]:


import grequests

class Test(object):
    def __init__(self, urls):
        self.urls = urls

    def exception(self, request, exception):
        print("Problem: {}: {}".format(request.url, exception))

    def async(self):
        results = grequests.map((grequests.get(u, timeout=10, stream=False) for u in self.urls), exception_handler=self.exception, size=200)
        # print(results)
        return results

# test = Test(news_data.news_url.values.tolist()[0:100])
# results = test.async()
# results


# In[70]:


import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

content_list = []
for i in tqdm(list(chunks(range(0, len(news_data.news_url.values)), 1000))):
    urls = news_data.news_url.values[i]
    test = Test(urls)
    results = test.async()
    for req in tqdm(results, leave=False):
        if req == None:
            content_list.append(None)
        elif req.status_code == 200:
            soup = BeautifulSoup(req.content, 'html.parser', from_encoding="iso-8859-1")
            req.close()
            arr = []
            for element in soup.find_all("p"):
                arr.append(element.getText())
            content_list.append(" ".join(arr))
        else:
            content_list.append(None)


# In[72]:


# content_list


# In[ ]:


news_data["content"] = content_list
news_data.to_csv("./new/content/news_data_with_content_{}_{}.tsv".format(start, end))


# In[48]:


# import requests
# url = "https://www.theage.com.au/world/europe/who-approves-pfizer-biontech-vaccine-for-emergency-use-20210101-p56r63.html"
# req = requests.get(url, timeout=5)
# soup = BeautifulSoup(req.content, 'html.parser')
# soup.find_all("p")


# In[14]:


# import requests
# from bs4 import BeautifulSoup
# def extract_text_from_url(url):
#     try:
#         # url = 'https://www.mopo.de/shoppingwelt/raclette-grill-fonduelette-mini-pizzaofen-33588944'
#         req = requests.get(url, timeout=5)
#         soup = BeautifulSoup(req.content, 'html.parser')
#         # print(soup.prettify())
# #         raw_text = []
# #         for element in soup.findAll(text=True):
# #             raw_text.append(element.getText())
#         return " ".join(soup.find_all("p"))
#     except:
#         return None


# In[50]:


# news_data.iloc[0:10,:]["news_url"].apply(extract_text_from_url)


# In[49]:


# from tqdm import tqdm

# tqdm.pandas()
# news_data["content"] = news_data["news_url"].progress_apply(extract_text_from_url)


# In[ ]:


# news_data.to_csv(".new/content/news_data_with_content_{}_{}.csv".format(start, end))


# In[16]:


# pd.read_csv("./new/content/news_data_with_content_0_10.tsv").content.values
