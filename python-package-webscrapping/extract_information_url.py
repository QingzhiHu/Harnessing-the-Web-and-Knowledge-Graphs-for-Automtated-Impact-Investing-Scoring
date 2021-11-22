#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("./search_results/msci_url_esg.csv", index_col=0)
df = df.dropna()
df


# In[2]:


# examination of the first matched results needed
df_rank = df[df["rank"].isin([1,2,3])]
df_rank


# In[3]:


# !pip install requests bs4 colorama


# In[4]:


from threading import Thread
import functools

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


# In[5]:


# @timeout(3)
# def test(n):
#     print(f'Sleeping for {n} seconds')
#     time.sleep(n)
#     # Real code here.
#     return 'Done'

# # test(2)  # OK
# test(5)  # -> Causes timeout.


# In[6]:


## working
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import colorama

# init the colorama module
# in case cannot open the website through openurl
# anonmy = 0
# number of urls visited so far will be stored here
total_urls_visited = 0
import sys
sys.setrecursionlimit(100000)

def find_csr_pdf_webtext(url):
    colorama.init()
    GREEN = colorama.Fore.GREEN
    GRAY = colorama.Fore.LIGHTBLACK_EX
    RESET = colorama.Fore.RESET
    YELLOW = colorama.Fore.YELLOW
    # initialize the set of links (unique links)
    internal_urls = set()
    external_urls = set()

    @timeout(5)
    def beautifulsoup_function(url):
        soup = BeautifulSoup(requests.get(url).content, "html.parser", from_encoding="iso-8859-1")
        return soup


    def is_valid(url):
        """
        Checks whether `url` is a valid URL.
        """
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def get_all_website_links(url):
        """
        Returns all URLs that is found on `url` in which it belongs to the same website
        """
        global anonmy
        # all URLs of `url`
        urls = set()
        # domain name of the URL without the protocol
        domain_name = urlparse(url).netloc
        try:
            # urllib.request.urlopen(url,timeout=5)
            # anonmy = 0

            # soup = BeautifulSoup(requests.get(url).content, "html.parser")
            soup = beautifulsoup_function(url)
            # print("hi")
            for a_tag in soup.findAll("a", href=True):
                href = a_tag.attrs.get("href")
                if href == "" or href is None:
                    # href empty tag
                    continue
                # join the URL if it's relative (not absolute link)
                href = urljoin(url, href)
                parsed_href = urlparse(href)
                # remove URL GET parameters, URL fragments, etc.
                href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
                if not is_valid(href):
                    # not a valid URL
                    print("debug not valid", href)
                    continue
                if href in internal_urls:
                    # already in the set
                    continue
                if domain_name not in href:
                    # external link
                    if href not in external_urls:
                        # print(f"{GRAY}[!] External link: {href}{RESET}")
                        external_urls.add(href)
                    continue
                # print(f"{GREEN}[*] Internal link: {href}{RESET}")
                urls.add(href)
                internal_urls.add(href)
            return urls
        except:
            return []


    def crawl(url, max_urls=30):
        """
        Crawls a web page and extracts all links.
        You'll find all links in `external_urls` and `internal_urls` global set variables.
        params:
          max_urls (int): number of max urls to crawl, default is 30.
        """
        global total_urls_visited
        total_urls_visited += 1
#         print(f"{YELLOW}[*] Crawling: {url}{RESET}")
        links = get_all_website_links(url)
        for link in links:
            if total_urls_visited > max_urls:
                break
            try:
                crawl(link, max_urls=max_urls)
            except:
                print("reaching maximum recursion depth")

    # initialize the set of links (unique links)
    # url = 'https://www.3m.com/3M/en_US/sustainability-us/annual-report/'
    internal_urls = set()
    external_urls = set()
    crawl(url)
    internal_urls = list(internal_urls)
    internal_urls_pdf = []
    internal_urls_web = []
    external_urls_pdf = []
    external_urls_web = []

    for url in internal_urls:
        if ".pdf" in url:
            internal_urls_pdf.append(url)
        else:
            internal_urls_web.append(url)


    for url in external_urls:
        if ".pdf" in url:
            external_urls_pdf.append(url)
        else:
            external_urls_web.append(url)

    return internal_urls_pdf, internal_urls_web, external_urls_pdf, external_urls_web


# In[7]:


# internal_urls_pdf, internal_urls_web, external_urls_pdf, external_urls_web = find_csr_pdf_webtext('https://www.abbott.com/responsibility/sustainability/sustainability-reporting/current-reports.html')


# In[8]:


# external_urls_pdf


# In[9]:


links = df_rank.url.values.tolist()


# In[10]:


len(links)


# In[11]:


# links


# In[ ]:


internal_pdfs = []
internal_web_texts = []

external_pdfs = []
external_web_texts = []

count = 0
total_urls_visited = 0

# import sys
# start=int(sys.argv[1])
# end=int(sys.argv[2])


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="starting index", default=0)
parser.add_argument("--end", help="ending index", default=len(links))
args = parser.parse_args()

start = args.start
end = args.end

print("start: ", start, " end: ", end)
for link in links[start:end]:
    count += 1
    if count % 1 == 0:
        print("in progress", count)
#     if count % 10 == 0 or count == len(links):
#         df_temp = pd.DataFrame()
#         df_temp["internal_pdfs"] = internal_pdfs
#         df_temp["internal_web_texts"] = internal_web_texts
#         df_temp["external_pdfs"] = external_pdfs
#         df_temp["external_web_texts"] = external_web_texts
#         df_temp["company"] = df_rank.company.values[start:end].tolist()
# #         df_temp.to_csv("esg_companies_"+str(count+1000)+".csv")
#         df_temp.to_csv("esg_companies_"+str(end)+".tsv", sep = '\t')
#         print("saved", end)

    total_urls_visited = 0 # in case something goes wrong
    internal_urls_pdf, internal_urls_web, external_urls_pdf, external_urls_web = find_csr_pdf_webtext(link)

    if internal_urls_pdf == []:
        if ".pdf" in link:
            internal_pdfs.append([link])
        else:
            internal_pdfs.append(None)
    else:
        internal_pdfs.append(internal_urls_pdf)

    if internal_urls_web == []:
        internal_web_texts.append(None)
    else:
        internal_web_texts.append(internal_urls_web)

    if external_urls_pdf == []:
        external_pdfs.append(None)
    else:
        external_pdfs.append(external_urls_pdf)

    if external_urls_web == []:
        external_web_texts.append(None)
    else:
        external_web_texts.append(external_urls_web)


df_temp = pd.DataFrame()
df_temp["internal_pdfs"] = internal_pdfs
df_temp["internal_web_texts"] = internal_web_texts
df_temp["external_pdfs"] = external_pdfs
df_temp["external_web_texts"] = external_web_texts
df_temp["company"] = df_rank.company.values[start:end].tolist()
#         df_temp.to_csv("esg_companies_"+str(count+1000)+".csv")
df_temp.to_csv("./company_reports_link/esg_companies_"+str(end)+".tsv", sep = '\t')
print("saved", end)


# In[ ]:


# df_temp = pd.DataFrame()
# df_temp["internal_pdfs"] = internal_pdfs
# df_temp["internal_web_texts"] = internal_web_texts
# df_temp["external_pdfs"] = external_pdfs
# df_temp["external_web_texts"] = external_web_texts
# df_temp["company"] = df_rank.company.values.tolist()
# df_temp.to_csv("esg_companies.csv")
