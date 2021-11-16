# Retrieve news
import urllib.request
from datetime import datetime

master_url = 'http://data.gdeltproject.org/gdeltv2/masterfilelist.txt'
master_file = urllib.request.urlopen(master_url)
min_date = datetime.strptime("2018 12 1 12", '%Y %m %d %H')
max_date = datetime.strptime("2019 1 1 12", '%Y %m %d %H')
to_download = []
for line in master_file:
    decoded_line = line.decode("utf-8")
    if 'gkg.csv.zip' in decoded_line:
        a = decoded_line.split(' ')
        file_url = a[2].strip()
        file_dte = datetime_object = datetime.strptime(file_url.split('/')[-1].split('.')[0], '%Y%m%d%H%M%S')
        if (file_dte > min_date and file_dte <= max_date):
            to_download.append(file_url)
print("{} file(s) to download since {}".format(len(to_download), min_date))

from zipfile import ZipFile
import gzip
import io
import os

def download_content(url, save_path):
    with urllib.request.urlopen(url) as dl_file:
        input_zip = ZipFile(io.BytesIO(dl_file.read()), "r")
        name = input_zip.namelist()[0]
        with gzip.open(save_path, 'wb') as f:
              f.write(input_zip.read(name))

def download_to_dbfs(url):
    file_name = '{}.gz'.format(url.split('/')[-1][:-4])
    tmp_file = '{}/{}'.format("./news2", file_name)
    download_content(url, tmp_file)
    # dbutils.fs.mv('file:{}'.format(tmp_file), 'dbfs:{}/{}'.format("/content", file_name))

n = len(to_download)
for i, url in enumerate(to_download):
    download_to_dbfs(url)
    print("{}/{} [{}]".format(i + 1, n, url))
