Firstly, one needs to provide a text file ./company_list/companies.txt for all the companies of interests. In the txt file, each company is separated by a new line. 

Currently there is a companies.txt inside of the folder ./company_list, and this file is generated through merging the 3 parquets files ./msci where the companies in ./msci/mcap_msci_em_approx_usd.parquet and ./msci/mcap_msci_world_approx_usd.parquet in the most recent 5 years and 1) not dead 2) have at least 50% data are selected. There are 2830 companies selected and the companies are written into ./company_list/companies.txt
 
The procedure of using msci to select companies of interests is written in a sample get_companies.ipynb for user's reference. User can also generate his own companies.txt and put it into the ./company_list directory. 

