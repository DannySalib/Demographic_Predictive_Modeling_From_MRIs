import pandas as pd 

print('Getting Participants Data Frame...')
print('__________________________________')
df_path = r'Data\participants.tsv'

try:
    df = pd.read_csv(df_path, sep='\t')
except FileNotFoundError as e:
    raise FileNotFoundError(f'{e}\nTry running `download_data.py` first!')

print(df.head())
print()

print('Class prioirs...')
print('________________')
classification_group_names = ['AgeGroup', 'Child_Adult', 'Gender']

for name in classification_group_names:
    print(f'Classification Name: {name}')
    for classification_group_name, df_classification_group in df.groupby(name):
        print(f'P(Y={classification_group_name}) = {len(df_classification_group) / len(df) * 100:.2f}%')
    print()