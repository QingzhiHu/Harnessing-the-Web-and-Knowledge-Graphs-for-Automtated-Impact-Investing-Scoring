#!/usr/bin/env python
# coding: utf-8

# In[1]:


# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]
#
# commands1 = []
# for i in list(chunks(range(0, 19327), 100)):
#     commands1.append("python .\download_data_sustainability_pdf.py --start {} --end {}".format(i[0], i[-1]))
# print(commands1)
#

# In[22]:


# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]
#
# commands2 = []
# for i in list(chunks(range(0, 1849223), 100000)):
#     commands2.append("python .\download_data_url.py --start {} --end {}".format(i[0], i[-1]))
# commands2
#

# In[2]:





# In[17]:


# import subprocess
# import os
# import time
#
# # commands = ["python .\download_data_url.py --start 0 --end 1000", "python .\download_data_url.py --start 1000 --end 2000"]
#
# processes = set()
# max_processes = 10
#
# for command in commands2:
#     processes.add(subprocess.Popen(command,shell=True))
#     while len(processes) >= max_processes:
#         time.sleep(.1)
#         processes.difference_update([
#             p for p in processes if p.poll() is not None])
#
# #Check if all the child processes were closed
# for p in processes:
#     if p.poll() is None:
#         p.wait()


# In[4]:


import subprocess
import os
import time

# commands = ["python .\download_data_url.py --start 0 --end 1000", "python .\download_data_url.py --start 1000 --end 2000"]

commands1 = ['wikimapper download nlwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download dewiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download eswiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download jawiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download frwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download plwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download svwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download ptwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download eowiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download nowiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download idwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download zhwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download fiwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download afwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download itwiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download kowiki-latest --dir D:\\wikipedia\\official\\mapper',
 'wikimapper download trwiki-latest --dir D:\\wikipedia\\official\\mapper']
processes = set()
max_processes = 20


for command in commands1:
    processes.add(subprocess.Popen(command,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
    while len(processes) >= max_processes:
        time.sleep(.1)
        processes.difference_update([
            p for p in processes if p.poll() is not None])

#Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()
