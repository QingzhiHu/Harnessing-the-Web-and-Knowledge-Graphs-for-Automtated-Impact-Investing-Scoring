# Web-scrapping ESG reports/texts

This repository contains a small package for automatically extracting ESG related text or pdfs from the company's website. 



## Environment set-up

```
pip install -r pip-requirements.txt
```

## Pipeline

The pipeline is as follows:

- 1. user needs to provide a companies.txt file where each company name is separated by a line, and put this file into ./company_list directory.
  
  2. run the following command ```python web_search.py ``` which will generate ./search_results/msci_url_esg.csv which contains the top 10 search results of "company_name + " company, official website sustainability report" offered by BingSearch v7 for every company (Note you will need a Microsoft Cognitive Account --> application BingSearch v7 to get your subscription_key and replaced it in the ./web_search.py).
  
  3. run the following command `python extract_information_url.py` which will read top 3 search results (urls) from the file ./search_results/msci_url_esg.csv for each company. Afterwards, for each url retrieved it will loop through the website and the links in the website to find 1) internal urls which are pdfs 2) external urls which are pdfs 3) internal urls which are webpage with text 4) external urls which are webpage with text. In this way, we can ensure most of the companies will have either pdfs or webpage info available for esg and sustainability related information. 
  
  4. It is recommende to run command `python extract_information_url.py--start 0 --end 1000` where you specify the starting and ending index of the url you want to process. In this way you can run the program parallel which makes it a lot faster. The results is save in "./company_reports_link/esg_companies_"+str(end)+".tsv".

For exampl, using the following command to run parallel to get the esg reports related text for each company:

```
python extract_information_url.py--start 0 --end 1000
python extract_information_url.py--start 1000 --end 2000
python extract_information_url.py--start 2000 --end 3000
```

## Some notes

Firstly, one needs to provide a text file ./company_list/companies.txt for all the companies of interests. In the txt file, each company is separated by a new line. 

Currently there is a companies.txt inside of the folder ./company_list, and this file is generated through merging the 3 parquets files ./msci where the companies in ./msci/mcap_msci_em_approx_usd.parquet and ./msci/mcap_msci_world_approx_usd.parquet in the most recent 5 years and 1) not dead 2) have at least 50% data are selected. There are 2830 companies selected and the companies are written into ./company_list/companies.txt

The procedure of using msci to select companies of interests is written in a sample get_companies.ipynb for user's reference. User can also generate his own companies.txt and put it into the ./company_list directory. 


