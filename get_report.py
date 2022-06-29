import sys
sys.path.append("/home/qingzhi/.conda/envs/test/lib/python3.9/site-packages")
import pandas as pd
df_pdf = pd.read_csv("./wiki_data/report_temp.csv")
df_pdf

import sys
sys.setrecursionlimit(50000)

import grequests

class Test(object):
    def __init__(self, urls):
        self.urls = urls

    def exception(self, request, exception):
        print("Problem: {}: {}".format(request.url, exception))

    def async1(self):
        results = grequests.map((grequests.get(u, timeout=10) for u in self.urls), exception_handler=self.exception, size=5)
        # print(results)
        return results

# test = Test(news_data.url.values.tolist()[0:100])
# results = test.async1()
# results
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

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin

from PyPDF2 import PdfFileReader
from io import BytesIO
import os
# @timeout(300)
# def process_pdf(req):
#     open_pdf_file = BytesIO(req.content)
#     req.close()
#     pdfFile = PdfFileReader(open_pdf_file, strict=False)
#     text = [pdfFile.getPage(i).extractText()
#             for i in range(0, pdfFile.getNumPages())]
#     return " ".join(text)
import fitz
@timeout(500)
def process_pdf(req):
    if os.path.exists("metadata.pdf"):
       os.remove("metadata.pdf")
    else:
      print("The file does not exist") 
    with open('metadata.pdf', 'wb') as f:
        f.write(req.content)
    doc = fitz.open("metadata.pdf")
    all_text = []
    for page in doc:
        text = page.getText()
        all_text.append(text)
    return " ".join(all_text)


from tqdm import tqdm
tqdm.pandas()
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

content_list = []
for i in tqdm(list(chunks(range(0, len(df_pdf.pdf_url.values)), 10))):
    urls = df_pdf.pdf_url.values[i]
    test = Test(urls)
    results = test.async1()
    # for req, url in tqdm(zip(results, urls), total=len(urls), leave=False):
    for req, url in tqdm(zip(results, urls), total=len(urls), leave=False):

        if req == None:
            content_list.append(None)
        elif req.status_code == 200:
            try:
                content_list.append(process_pdf(req))
                print(url)
            except:
                content_list.append(None)
                # print("pdf time out")
        else:
            content_list.append(None)

df_pdf["pdf_content"] = content_list
df_pdf = df_pdf.dropna()
df_pdf.to_csv("sus_reports_content_all.tsv", sep="\t")
# df_pdf["pdf_content"] = df_pdf["pdf_content"].apply(lambda x: np.nan if x==np.nan else str(x).encode('utf-8', 'replace').decode('utf-8'))
