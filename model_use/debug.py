import numpy as np
import pandas as pd
from Bio import SeqIO


def get_utr(file_path, type, id_col='none', utr_col='none'):
    seq_id = []
    seq_fa = []
    if type == 'fasta':
        for seq_record in SeqIO.parse(file_path, "fasta"):
            seq_id.append(str(seq_record.id))
            seq_fa.append(str(seq_record.seq))
        utr_dt = {'id': seq_id, 'utr': seq_fa}
        df_utr = pd.DataFrame(utr_dt)
        df_utr["len"] = df_utr["utr"].apply(lambda x: len(x))
    elif type in ['csv', 'txt']:
        if type == 'csv':
            df_pre = pd.read_csv(file_path)
        elif type == 'txt':
            df_pre = pd.read_table(file_path)
        utr_id = df_pre.loc[:, id_col]
        utr_seq = df_pre.loc[:, utr_col]
        utr_dt = {'id': utr_id, 'utr': utr_seq}
        df_utr = pd.DataFrame(utr_dt)
        df_utr["len"] = df_utr["utr"].apply(lambda x: len(x))

    return df_utr


def pad_left_N(df,seq_len=100):
    df_pad = df
    df_pad.loc[:, 'pad_left_N'] = df.loc[:, 'utr']
    for i in range(len(df_pad)):
        utr_len = df_pad.loc[i, 'len']
        if utr_len < seq_len:
            N_num = seq_len - utr_len
            df_pad.loc[i, 'pad_left_N'] = N_num*'N' + df_pad.loc[i, 'utr']
    return df_pad


def pad_right_N(df,seq_len=100):
    df_pad = df
    df_pad.loc[:, 'pad_right_N'] = df.loc[:, 'utr']
    for i in range(len(df_pad)):
        utr_len = df_pad.loc[i, 'len']
        if utr_len < seq_len:
            N_num = seq_len - utr_len
            df_pad.loc[i, 'pad_right_N'] = df_pad.loc[i, 'utr'] + N_num*'N'
    return df_pad


def pad_N(df, orient='left', seq_len=100):
    df_pad = df
    pad_col = 'pad_' + orient + '_N'
    df_pad.loc[:, pad_col] = df.loc[:, 'utr']
    if orient == 'left':
        for i in range(len(df_pad)):
            utr_len = df_pad.loc[i, 'len']
            if utr_len < seq_len:
                N_num = seq_len - utr_len
                df_pad.loc[i, pad_col] = N_num*'N' + df_pad.loc[i, 'utr']
    elif orient == 'right':
        for i in range(len(df_pad)):
            utr_len = df_pad.loc[i, 'len']
            if utr_len < seq_len:
                N_num = seq_len - utr_len
                df_pad.loc[i, pad_col] = df_pad.loc[i, 'utr'] + N_num*'N'
        

