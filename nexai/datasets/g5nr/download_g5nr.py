import os

import urllib3
import pandas

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('output', metavar='G5NR_7km/', type=str,
                    help='Where to save your data?')

args = parser.parse_args()

def url_to_pandas(url):
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    return pandas.read_html(r.data)[-2]

path = 'https://portal.nccs.nasa.gov/datashare/gmao_obsteam/osse_for_wisc/'
to_directory = args.output

if not os.path.exists(to_directory):
    os.makedirs(to_directory)

df = url_to_pandas(path)
files = []

for name in df['Name']:
    try:
        int(name[:8])
    except:
        continue

    folder_path = os.path.join(path, name)
    df_folder = url_to_pandas(folder_path)
    for filename in df_folder['Name']:
        if isinstance(filename, str):
            if 'nc4' == filename[-3:]:
                filepath = os.path.join(folder_path, filename)
                print(f"Filepath: {filepath}")
                if os.path.exists(filepath):
                    continue
                    
                wget_cmd = f"wget -O {to_directory}/{filename} {filepath}"
                os.system(wget_cmd)