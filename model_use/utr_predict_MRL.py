## Imports
from importlib import reload
from pathlib import Path
import random
random.seed(1337)
import os, sys
from decimal import Decimal

# numpy and similar
import numpy as np
np.random.seed(1337)
import pandas as pd
pd.options.mode.chained_assignment = None 
import scipy.stats as stats

# Dont use GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Deep Learning packages
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model
from sklearn import preprocessing

# code scripts
import model
import utils
import utils_data
from Bio import SeqIO

## change work path
os.chdir(sys.path[0])


## 1.notice: for optimus model
##   Varying_length_25to100_model's "seq_len_limit" should set 100
##   main_MRL_model's "seq_len_limit" should set 50
##   retrained_evolution_model's "seq_len_limit" should set 54
##
## 2.If the input file is in csv or txt format, the "id_col" is the ID column name of utr, 
##   the "utr_col" is the column name of the utr sequence
##
## 3.model list
##   optimus_model_dict = {
##       1: 'Varying_length_25to100_model',
##       2: 'main_MRL_model',
##       3: 'retrained_evolution_model'}
## 
##   frame_pool_model_dict = {
##       1: 'utr_model_combined_residual',
##       2: 'utr_model_combined_residual_noTG',
##       3: 'utr_model_combined_residual_uORF'}


## Converting utr sequence to one-pot matrix 
def one_hot_encode(df, col='utr', orient='right', seq_len=100):
    nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    vectors=np.empty([len(df),seq_len,4])
    if orient == 'right':
        for i,seq in enumerate(df[col].str[-seq_len:]): 
            seq = seq.lower()
            a = np.array([nuc_d[x] for x in seq])
            vectors[i] = a
    elif orient == 'left':
        for i,seq in enumerate(df[col].str[:seq_len]): 
            seq = seq.lower()
            a = np.array([nuc_d[x] for x in seq])
            vectors[i] = a
    
    return vectors


## Get utr sequence from input file
def get_utr(file_path, type, id_col='id', utr_col='utr'):
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
            df_pre = pd.read_csv(file_path, sep='\t')
        utr_id = df_pre.loc[:, id_col]
        utr_seq = df_pre.loc[:, utr_col]
        utr_dt = {'id': utr_id, 'utr': utr_seq}
        df_utr = pd.DataFrame(utr_dt)
        # df_utr["len"] = df_utr["utr"].apply(lambda x: len(x))
        # df_utr = df_pre

    return df_utr


## For optimus model, padding N for utr shorter than the limit length
def pad_N(df, orient='left', seq_len=100):
    df_pad = df
    pad_col = 'pad_' + orient + '_N'
    df_pad.loc[:, pad_col] = df.loc[:, 'utr']
    if orient == 'left':
        # for i in range(len(df_pad)):
        #     utr_len = df_pad.loc[i, 'len']
        #     if utr_len < seq_len:
        #         N_num = seq_len - utr_len
        #         df_pad.loc[i, pad_col] = N_num*'N' + df_pad.loc[i, 'utr']

        df_pad.loc[:, pad_col] = 100*'N' + df_pad.loc[:, 'utr']
    elif orient == 'right':
        # for i in range(len(df_pad)):
        #     utr_len = df_pad.loc[i, 'len']
        #     if utr_len < seq_len:
        #         N_num = seq_len - utr_len
        #         df_pad.loc[i, pad_col] = df_pad.loc[i, 'utr'] + N_num*'N'
        
        df_pad.loc[:, pad_col] = df_pad.loc[:, 'utr'] + 100*'N'
    return df_pad, pad_col


## Predict MRL using optimus model
def pred_mrl_optimus(df, model_name='Varying_length_25to100_model', pad_orient='left', one_hot_orient='right', len=100, out_suffix=''):
    model_dict = {
        'main_MRL_model': 'model/optimus/main_MRL_model.hdf5',
        'Varying_length_25to100_model': 'model/optimus/Varying_length_25to100_model.hdf5',
        'retrained_evolution_model': 'model/optimus/retrained_evolution_model.hdf5'
    }
    # pad_col = pad_col
    # one_hot_orient = one_hot_orient
    model_path = model_dict[model_name]
    model = keras.models.load_model(model_path)
    df_know = pd.read_csv('data/optimus/GSM3130435_egfp_unmod_1.csv')
    df_know.sort_values('total_reads', ascending=False).reset_index(drop=True)
    scale_utrs = df_know[:40000]
    scaler = preprocessing.StandardScaler()
    scaler.fit(scale_utrs['rl'].values.reshape(-1,1))

    df_pad, pad_col = pad_N(df, orient=pad_orient, seq_len=len)
    utr_to_vector = one_hot_encode(df_pad, col=pad_col, orient=one_hot_orient, seq_len=len)
    pred_optimus = model.predict(utr_to_vector).reshape(-1, 1)

    out_col_name = 'pred_optimus' + out_suffix
    df_pad.loc[:,out_col_name] = scaler.inverse_transform(pred_optimus)
    
    return df_pad, out_col_name


## Predict MRL using frame pool model
def pred_mrl_frame_pool(df, model_name='utr_model_combined_residual', col_name='utr', out_suffix=''):
    model_dict = {
        'utr_model_combined_residual': 'model/frame_pool/utr_model_combined_residual.h5',
        'utr_model_combined_residual_noTG': 'model/frame_pool/utr_model_combined_residual_noTG.h5',
        'utr_model_combined_residual_uORF': 'model/frame_pool/utr_model_combined_residual_uORF.h5'
    }
    model_path = model_dict[model_name]
    utr_model = load_model(model_path, custom_objects={'FrameSliceLayer': model.FrameSliceLayer})
    one_hot_fn_utr = utils_data.OneHotEncoder(col_name, min_len=None)
    library_fn = utils_data.LibraryEncoder("library", {"egfp_unmod_1":0, "random":1})
    gen_utr = utils_data.DataSequence(df, encoding_functions=[one_hot_fn_utr] + [library_fn], shuffle=False)
    pred_frame_pool = utr_model.predict_generator(gen_utr, verbose=0)
    out_col_name = 'pred_frame_pool' + out_suffix
    df.loc[:,out_col_name] = pred_frame_pool
    
    return df, out_col_name
    

## Output MRL prediction result
def out_mrl_pred(file_in, file_type, optimus_model=1, frame_pool_model=1, pad_orient='left', one_hot_orient='right', seq_len_limit=100, id_col='id', utr_col='utr', out_suffix=''):
    file_in_path = 'input/' + file_in
    file_out_path = 'output/' + str(file_in).split('.')[0] + '_pred_mrl.csv'
    
    optimus_model_dict = {
        1: 'Varying_length_25to100_model',
        2: 'main_MRL_model',
        3: 'retrained_evolution_model'
    }
    frame_pool_model_dict = {
        1: 'utr_model_combined_residual',
        2: 'utr_model_combined_residual_noTG',
        3: 'utr_model_combined_residual_uORF'
    }

    optimus_model_select = optimus_model_dict[optimus_model]
    frame_pool_model_select = frame_pool_model_dict[frame_pool_model]

    df_utr = get_utr(file_in_path, file_type, id_col=id_col, utr_col=utr_col)
    df_pad, col_optimus = pred_mrl_optimus(df_utr, model_name=optimus_model_select, pad_orient=pad_orient, one_hot_orient=one_hot_orient, len=seq_len_limit, out_suffix=out_suffix)
    df, col_frame_pool = pred_mrl_frame_pool(df_pad, model_name=frame_pool_model_select, col_name='utr', out_suffix=out_suffix)
    
    if file_type in ['csv', 'txt']:
        if file_type == 'csv':
            df_raw = pd.read_csv(file_in_path)
        elif file_type == 'txt':
            df_raw = pd.read_csv(file_in_path, sep='\t')
        df_raw.loc[:,col_optimus] = df.loc[:,col_optimus]
        df_raw.loc[:,col_frame_pool] = df.loc[:,col_frame_pool]
        df_raw.to_csv(file_out_path, index=False)
    elif file_type == 'fasta':
        df.to_csv(file_out_path, index=False)
    print(file_in + ' predict MRL result in: ' + file_out_path)


## test example
# out_mrl_pred('rh_utr5.fasta', file_type='fasta')
# out_mrl_pred('test_utr.txt', file_type='txt', id_col='id', utr_col='utr')
# out_mrl_pred('pred_why_utr.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='name', utr_col='seq', seq_len_limit=100, out_suffix='_SB')
# out_mrl_pred('all_utr5.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='utr200', seq_len_limit=100, out_suffix='_200')
# out_mrl_pred('all_utr_test.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='utr100', seq_len_limit=100, out_suffix='_100')
# out_mrl_pred('utr_paper.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='utr', seq_len_limit=100, out_suffix='_paper')
# out_mrl_pred('utr5design.txt', file_type='txt', id_col='id', utr_col='utr')
# out_mrl_pred('原生eGFP_5UTR.fasta', file_type='fasta')
# out_mrl_pred('utr_paper_selected20.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='utr', seq_len_limit=100, out_suffix='')
# out_mrl_pred('all_utr5_dedup.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='utr', seq_len_limit=100, out_suffix='_100')
# out_mrl_pred('all_utr5_less100.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='utr', seq_len_limit=100, out_suffix='')
# out_mrl_pred('all_utr5_dedup_kozak.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='utr', seq_len_limit=100, out_suffix='')
# out_mrl_pred('seq.txt', file_type='fasta')
# out_mrl_pred('xupan_5utr_2.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='Ttr', seq_len_limit=100, out_suffix='')
# out_mrl_pred('MRL_diffusion.txt', file_type='fasta')
# out_mrl_pred('All_promoter_fragments_Native - 副本.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='Unnamed: 0', utr_col='seq110', seq_len_limit=100, out_suffix='')
# out_mrl_pred('3merRF_designed_5UTR.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='UTR5', seq_len_limit=100, out_suffix='')
# out_mrl_pred('test1.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='Sequence', seq_len_limit=100, out_suffix='')
# out_mrl_pred('test1_pred_mrl_selected_in.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col='id', utr_col='Sequence', seq_len_limit=100, out_suffix='')
out_mrl_pred('result_to_download.csv', optimus_model=1, frame_pool_model=1, file_type='csv', id_col="name", utr_col="5' UTRs", seq_len_limit=100, out_suffix='')

