import pandas as pd
import progressbar as pb

msnbc_dataframe = pd.DataFrame(dict((str(i), []) for i in range(1, 18)))

with open('msnbc990928.seq') as f:
    msnbc_file = f.readlines()

with pb.ProgressBar(max_value=989825) as bar:
    index = 0
    for line in msnbc_file:
        index += 1
        sequence = line.rstrip().split()
        if len(sequence)!=0 and sequence[0].isdigit():
            temp_dict = {'1':0, '2':0, '3':0, '4':0, '5':0,
                     '6':0, '7':0, '8':0, '9':0, '10':0,
                     '11':0, '12':0, '13':0, '14':0, '15':0,
                     '16':0, '17':0}
            for element in sequence:
                temp_dict[element] += 1
            msnbc_dataframe = msnbc_dataframe.append(temp_dict, ignore_index=True)
        bar.update(index)

msnbc_dataframe.to_csv('msnbc.csv', sep='\t', encoding='utf-8')
