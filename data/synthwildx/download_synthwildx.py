#!/usr/bin/env python

import os
import pandas
import urllib.request
from urllib.error import HTTPError
from tqdm import tqdm
from time import sleep
headers = {'User-Agent' : 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7',}

tab = pandas.read_csv('list.csv')
for index in tqdm(tab.index):
    url = tab.loc[index,'url']
    fpath = tab.loc[index,'filename']
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    while not os.path.isfile(fpath):
        try:
            request_site = urllib.request.Request(url, None, headers)
            with urllib.request.urlopen(request_site) as response:
                with open(fpath, 'wb') as fid:
                    fid.write(response.read())
            sleep(0.01)
        except KeyboardInterrupt as err:
            raise err # quit
        except HTTPError:
            print('HTTPError', url)
            break
        except:
            print('sleep', flush=True)
            sleep(1.0)