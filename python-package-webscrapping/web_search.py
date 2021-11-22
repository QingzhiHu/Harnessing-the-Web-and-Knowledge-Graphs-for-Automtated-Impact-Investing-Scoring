#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install beautifulsoup4')
# get_ipython().system('pip install urllib3')
# get_ipython().system('pip install pandas')
# get_ipython().system('pip install numpy')


# In[22]:


with open("./company_list/companies.txt", "r") as file:
    lines = file.readlines()
    companies_list = [line.rstrip() for line in lines]


# In[23]:


# # filter out dead companies
# companies_list = [x for x in companies_list if "dead" not in x.lower()]
len(companies_list)


# In[51]:


# # alternative companies S&P 500
# import pandas as pd

# # There are 2 tables on the Wikipedia page
# # we want the first table

# payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# first_table = payload[0]
# second_table = payload[1]

# df = first_table
# companies_list = df.Security.values.tolist()
# # companies_symbol_list = df.Symbol.value.tolist()


# In[49]:





# In[50]:


import requests
from bs4 import BeautifulSoup
import urllib3
import re
import pandas as pd

def top_search_result_esg_company(company_name):
  subscription_key = "c5a282b2cba14ce5a3ad451c3575bd91"
  search_url = "https://api.bing.microsoft.com/v7.0/search"
  search_term = company_name + " company, official website sustainability report"
  headers = {"Ocp-Apim-Subscription-Key": subscription_key}
  params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "mkt":"en-US"}
  response = requests.get(search_url, headers=headers, params=params)
  response.raise_for_status()
  search_results = response.json()
  rows = "\n".join(["""<tr>
                       <td><a href=\"{0}\">{1}</a></td>
                       <td>{2}</td>
                     </tr>""".format(v["url"], v["name"], v["snippet"])
                  for v in search_results["webPages"]["value"]])

  soup = BeautifulSoup("<table>{0}</table>".format(rows))
  links = []
  for link in soup.findAll('a'):
      links.append(link.get('href'))
  
  df = pd.read_html("<table>{0}</table>".format(rows))[0].rename(columns={0: "site", 1: "summary"})
  df["url"] = links
  df["rank"] = range(1, len(df)+1)
  df["company"] = company_name
  return df


# In[55]:


df_list = []
count = 0
for company in companies_list:
  count += 1
  if count%20 == 0:
    print("progress", count)
  df = top_search_result_esg_company(company)
  df_list.append(df)
df_merged = pd.concat(df_list)


# In[58]:


df_merged.to_csv("./search_results/msci_url_esg.csv")


# In[39]:


# df1 = top_search_result_esg_company(companies_list[0])


# In[40]:


# df2 = top_search_result_esg_company("Facebook")


# In[28]:



# subscription_key = "c5a282b2cba14ce5a3ad451c3575bd91"
# search_url = "https://api.bing.microsoft.com/v7.0/search"
# search_term = companies_list[1] + " company official website sustainability report"
# headers = {"Ocp-Apim-Subscription-Key": subscription_key}
# params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
# response = requests.get(search_url, headers=headers, params=params)
# response.raise_for_status()
# search_results = response.json()
# rows = "\n".join(["""<tr>
#                       <td><a href=\"{0}\">{1}</a></td>
#                       <td>{2}</td>
#                     </tr>""".format(v["url"], v["name"], v["snippet"])
#                 for v in search_results["webPages"]["value"]])

# soup = BeautifulSoup("<table>{0}</table>".format(rows))
# links = []
# for link in soup.findAll('a'):
#     links.append(link.get('href'))

# df = pd.read_html("<table>{0}</table>".format(rows))[0].rename(columns={0: "site", 1: "summary"})
# df["url"] = links
# df["rank"] = range(1, len(df)+1)
# df["company"] = companies_list[1]
# df

