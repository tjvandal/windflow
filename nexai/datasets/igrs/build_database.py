import os, sys
import glob

import pandas as pd

import sqlite3

igra_directory = '/nobackupp10/tvandal/data/IGRA/'

# ['data-y2d', 'igra2-readme.txt', 'igra2-country-list.txt', 'igra2-station-list.txt', 'status.txt',
# 'igra2-list-format.txt', 'igra2-data-format.txt', 'igra2-us-states.txt', 'igra2-readme.txt.1']

#con = sqlite3.connect("data/portal_mammals.sqlite")
# Load the data into a DataFrame
#surveys_df = pd.read_sql_query("SELECT * from surveys", con)

# Select only data for 2002
#surveys2002 = surveys_df[surveys_df.year == 2002]

# Write the new DataFrame to a new SQLite table
#surveys2002.to_sql("surveys2002", con, if_exists="replace")
#con.close()

class RawIGRA:
    def __init__(self, directory):
        self.dir = directory
        

    def stations(self):
        path = os.path.join(self.dir, 'igra2-station-list.txt')
        #names = ['id', 'lat', 'lon', 'elevation', 'state', 'name', 'first_year', 'last_year', 'n_obs']
        #df = pd.read_table(path, sep='\t')
        data = []
        with open(path, 'r') as reader:
            for line in reader.readlines():
                station = {'id': line[:11], 'lat': float(line[12:20]), 'lon': float(line[21:30]),
                        'elevation': float(line[31:37]), 'state': line[38:40].strip(), 'name': line[41:71].strip(),
                        'first_year': int(line[72:76]), 'last_year': int(line[77:81]), 
                        'n_obs': int(line[82:88])}
                data.append(station)
        data = pd.DataFrame(data)
        return data
    
    def station_data(self, station):
        path = os.path.join(self.dir, f'data-y2d/{station}-data.txt')
        if not os.path.exists(path):
            print(f"Cannot find data for station {station}")
            return
        print(f"Found data for stations {station}")
        data = []
        
        with open(path, 'r') as reader:
            for i, line in enumerate(reader.readlines()):
                if line[0] == '#': # header row
                    record_header = {'station': station, 'year': int(line[13:17]), 'month': int(line[18:20]), 
                          'day': int(line[21:23]), 'hour': int(line[24:26]),
                          'reltime': int(line[27:31]), 'num_levels': int(line[32:36]),
                          'p_src': line[37:45].strip(), 'np_src': line[46:54].strip(), 
                          'lat': int(line[55:62])/10000., 'lon': int(line[63:71])/10000.,
                         }
                    #print(record_header)
                else:
                    record = {'level_type1': int(line[0]), 'level_type2': int(line[1]), 'elapsed_time': int(line[3:8]),
                              'pressure': int(line[9:15]), 'pflag': line[15], 'geopotential_height': int(line[16:21]), 
                              'zflag': line[21], 'temp': int(line[22:27]), 'tflag': line[27], 'rh': int(line[28:33]),
                              'dpdp': int(line[34:39]), 'wind_direction': int(line[40:45]), 'wind_speed': int(line[46:51])
                             }
                    record.update(record_header)
                    data.append(record)
               
        data = pd.DataFrame(data)
        return data
    
    
def write_to_database():  
    conn = sqlite3.connect(os.path.join(igra_directory, 'igra.db'))
    
    igra = RawIGRA(igra_directory)
    stations = igra.stations()
    stations.to_sql("stations", conn, if_exists="replace")
    alldata = []
    for i, row in stations.iterrows():
        station_data = igra.station_data(row.id)
        if station_data is not None:
            station_data.to_sql("records", conn, if_exists="append", index=False)
            conn.commit()
    
def query_records():
    conn = sqlite3.connect(os.path.join(igra_directory, 'igra.db'))
    df = pd.read_sql_query("SELECT * from records", conn)
    print(df)
    
if __name__ == '__main__':
    write_to_database()
    query_records()