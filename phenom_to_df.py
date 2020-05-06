import pandas as pd
import os

def csv_to_df(directory):
    data = []    
    for fname in os.scandir(directory):
        with open(fname, 'r') as file:
            for line in file.readlines():
                line = line.replace(';', ',')
                line = line.rstrip(',\n')
                line = line.split(',')
                data.append(line)
    
    #Find the longest line in the data 
    longest_line = max(data, key = len)
        
    #Set the maximum number of columns
    max_col_num = len(longest_line)

    #Set the columns names
    col_names = ['event_ID', 'process_ID', 'event_weight', 'MET', 'MET_Phi']

    for i in range(1, (int((max_col_num-5)/5))+1):
        col_names.append('obj'+str(i))
        col_names.append('E'+str(i))
        col_names.append('pt'+str(i))
        col_names.append('eta'+str(i))
        col_names.append('phi'+str(i))

    #Create a dataframe from the list, using the column names from before
    df = pd.DataFrame(data, columns=col_names)
    df.fillna(value=pd.np.nan, inplace=True)
    
    #Pickle the dataframe to keep it fresh
    p_path = 'my_df.pkl'
    df.to_pickle(p_path)
    
    return df
