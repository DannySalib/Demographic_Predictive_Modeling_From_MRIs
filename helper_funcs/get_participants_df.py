import pandas as pd

from helper_funcs import download_file

PARTICIPANTS_DF_FILE_NAME: str = 'participants.tsv'
PARTICIPANTS_DF_KEY = 'ds000228/participants.tsv'
PARTICIPANTS_DF_PATH = './Data'

def download_participants_df() -> None:
    '''Download the particpants table'''
    return download_file(
        key = PARTICIPANTS_DF_KEY,
        the_file_name_you_want = PARTICIPANTS_DF_FILE_NAME,
        path = PARTICIPANTS_DF_PATH
    )

def get_participants_df() -> pd.DataFrame:
    '''read participants table as data frame'''
    path = download_participants_df()
    assert path == PARTICIPANTS_DF_PATH, f'Cannot get participants df. Unexpected path: \"{path}\"'
    return pd.read_csv(path)

