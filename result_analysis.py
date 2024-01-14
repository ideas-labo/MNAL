import argparse
import pdb
import pickle

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

from sklearn.metrics import (accuracy_score,recall_score,precision_score,f1_score, auc, roc_curve, confusion_matrix)


parser = argparse.ArgumentParser()
parser.add_argument('--table')
parser.add_argument('--fig')
parser.add_argument('--method_settings', nargs='+')
args = parser.parse_args()

sample_times = int(args.run)
step = 10

if args.table != None:
    table = int(args.table)
else:
    table = 0
if args.fig != None:
    fig = int(args.fig)
else:
    fig = 0
    
method_settings = list(args.method_settings)


def get_flags():
    if method_setting=='random':
        # random
        f1_flag = 'no' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'random'

    elif method_setting=='certainty':
        # certainty
        f1_flag = 'certainty' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single'

    elif method_setting=='iden':
        # read
        f1_flag = 'read' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single'

    elif method_setting=='iden':
        # iden
        f1_flag = 'iden' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single'

    elif method_setting=='Dominant':
        # Dominant(certainty+read+iden)
        f1_flag = 'certainty' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'read'
        f5_flag = 'iden'
        MODE = 'dominated'

    elif method_setting=='normalized_sum':
        # normalized_sum
        f1_flag = 'no' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'normalized_sum'
        f5_flag = 'no'
        MODE = 'normalized_sum'

    elif method_setting=='Knee':
        # Knee(certainty+read+iden)
        f1_flag = 'certainty' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'read'
        f5_flag = 'iden'
        MODE = 'knee'

    elif method_setting=='kmeans(normalized_sum)':
        # kmeans(normalized_sum)
        f1_flag = 'certainty' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'kmeans'

    elif method_setting=='labeled_diversity':
        # labeled diversity
        f1_flag = 'no' # 'certainty' or 'no'
        f2_flag = 'labeled' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single'

    elif method_setting=='unlabeled diversity':
        # unlabeled diversity
        f1_flag = 'no' # 'certainty' or 'no'
        f2_flag = 'unlabeled' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single'

    elif method_setting=='labeled representative':
        # labeled representative
        f1_flag = 'no' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'labeled' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single'

    elif method_setting=='unlabeled representative':
        # unlabeled representative
        f1_flag = 'no' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'minus_unlabeled' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single'

    elif method_setting=='BALD':
        # BALD
        f1_flag = 'no' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'BALD' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single'

    elif method_setting=='Coreset':
        # Coreset
        f1_flag = 'no' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'coreset' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single'

    elif method_setting=='threshold':
        # threshold
        f1_flag = 'threshold' # 'certainty' or 'no'
        f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
        f4_flag = 'no'
        f5_flag = 'no'
        MODE = 'single' 
        threshold = 0.1
    return f1_flag, f2_flag, f3_flag, f4_flag, f5_flag, MODE


if table == 1 or fig == 7:
    result = np.zeros((3,100,7,3))
    for idx, method_setting in enumerate(['random','certainty','normalized_sum']):
        f1_flag, f2_flag, f3_flag, f4_flag, f5_flag, MODE = get_flags()
        for i,size in enumerate([300,500,700]):
            result2 = np.zeros((sample_times,3))
            temp = np.zeros((7,10,10))
            for run in range(1, sample_times+1):
                for temp_step in range(1, step+1):
                    with open(f'/rq1/data/rq1_initial_size{size}_sample_size{size}_run_{run}_step{temp_step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}.pkl','rb') as pklfile:
                        files = pickle.load(pklfile)
                    _, _, _, metrics_update, saved_indices, _, read_values, iden_values  = files
                    temp[5][run-1][temp_step-1] = np.average(read_values)
                    temp[6][run-1][temp_step-1] = np.average(iden_values)

                with open(f'/rq1/data/rq1_initial_size{size}_sample_size{size}_run_{run}_step{10}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}.pkl','rb') as pklfile:
                    files = pickle.load(pklfile)
                _, _, _, metrics_update, saved_indices, _, read_values, iden_values  = files

                for temp_step in range(1,11):
                    for j in range(5):
                        temp[j][run-1][temp_step-1] += metrics_update[j][temp_step]

            print(f'{method_setting}, query size {size}')

            # table 1
            if table == 1:
                print(f'&&{round(np.average(temp[0]),3)}&&{round(np.average(temp[1]),3)}&&{round(np.average(temp[3]),3)}&&{round(np.average(temp[4]),3)}&&{round(np.average(temp[5]),3)}&&{round(np.average(temp[6]),3)}\\\\')
                print(f'&&{round(np.std(temp[0]),3)}&&{round(np.std(temp[1]),3)}&&{round(np.std(temp[3]),3)}&&{round(np.std(temp[4]),3)}&&{round(np.std(temp[5]),3)}&&{round(np.std(temp[6]),3)}\\\\')

            if fig == 7:
            # fig 7
                input_11 = [temp_var for temp_var in np.average(temp[0],axis=0)]
                input_12 = [temp_var for temp_var in np.std(temp[0],axis=0)]
                input_21 = [temp_var for temp_var in np.average(temp[5],axis=0)]
                input_22 = [temp_var for temp_var in np.std(temp[5],axis=0)]
                input_31 = [temp_var for temp_var in np.average(temp[6],axis=0)]
                input_32 = [temp_var for temp_var in np.std(temp[6],axis=0)]
                files = input_11, input_12, input_21, input_22, input_31, input_32
                with open(f'/rq1_{method_setting}_{size}','wb') as pklfile:
                    pickle.dump(files, pklfile)
            for temp1 in range(7):
                result[idx,:,temp1,i] = temp[temp1].flatten()

    # scott-knott
    for size in [300,500,700]:
        result = np.zeros((sample_times,7,3))
        for i,method_setting in enumerate(['random','certainty','normalized_sum']):
            f1_flag, f2_flag, f3_flag, f4_flag, f5_flag, MODE = get_flags()
            for run in range(1, sample_times+1):
                with open(f'/rq1/data/rq1_initial_size{size}_sample_size{size}_run_{run}_step{step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}.pkl','rb') as pklfile:
                    files = pickle.load(pklfile)
                _, _, _, metrics_update, saved_indices, _, read_values, iden_values  = files

                result[run-1][5][i] += np.average(read_values)
                result[run-1][6][i] += np.average(iden_values)
                for j in range(5):
                    result[run-1][j][i] += metrics_update[j][-1]
        
        result2 = np.zeros((sample_times,3))
        for i in range(7):
            result2[:,0] = result[:,i,0]
            result2[:,1] = result[:,i,1]
            result2[:,2] = result[:,i,2]
            df = pd.DataFrame(result2)
            df.to_csv(f"skdata_{i}_size_{size}.csv",index=False,encoding='utf-8-sig')


if table == 2 or fig == 8:
    result = np.zeros((2,100,7,3))
    for idx,method_setting in enumerate(['normalized_sum','normalized_sum']):
        f1_flag, f2_flag, f3_flag, f4_flag, f5_flag, MODE = get_flags()
        for i,size in enumerate([300,500,700]):
            result2 = np.zeros((sample_times,3))
            temp = np.zeros((7,10,10))
            for run in range(1, sample_times+1):
                if idx == 0:
                    for temp_step in range(1, step+1):
                        with open(f'/rq1/data/rq1_initial_size{size}_sample_size{size}_run_{run}_step{temp_step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}.pkl','rb') as pklfile:
                            files = pickle.load(pklfile)
                        _, _, _, metrics_update, saved_indices, _, read_values, iden_values  = files
                        temp[5][run-1][temp_step-1] = np.average(read_values)
                        temp[6][run-1][temp_step-1] = np.average(iden_values)

                    with open(f'/rq1/data/rq1_initial_size{size}_sample_size{size}_run_{run}_step{10}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}.pkl','rb') as pklfile:
                        files = pickle.load(pklfile)
                    _, _, _, metrics_update, saved_indices, _, read_values, iden_values  = files
                else:
                    for temp_step in range(1, step+1):
                        with open(f'/rq2/data/rq2_initial_size{size}_sample_size{size}_run_{run}_step{temp_step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_ratio1.pkl','rb') as pklfile:
                            files = pickle.load(pklfile)
                        _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files
                        temp[5][run-1][temp_step-1] = np.average(read_values)
                        temp[6][run-1][temp_step-1] = np.average(iden_values)

                    with open(f'/rq2/data/rq2_initial_size{size}_sample_size{size}_run_{run}_step{10}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_ratio1.pkl','rb') as pklfile:
                        files = pickle.load(pklfile)
                    _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files

                for temp_step in range(1,11):
                    for j in range(5):
                        temp[j][run-1][temp_step-1] += metrics_update[j][temp_step]
            print(f'{method_setting}, query size {size}')

            # table 2
            if table == 2:
                print(f'&&{round(np.average(temp[0]),3)}&&{round(np.average(temp[1]),3)}&&{round(np.average(temp[3]),3)}&&{round(np.average(temp[4]),3)}&&{round(np.average(temp[5]),3)}&&{round(np.average(temp[6]),3)}\\\\')
                print(f'&&{round(np.std(temp[0]),3)}&&{round(np.std(temp[1]),3)}&&{round(np.std(temp[3]),3)}&&{round(np.std(temp[4]),3)}&&{round(np.std(temp[5]),3)}&&{round(np.std(temp[6]),3)}\\\\')
                
            # fig 8
            if fig == 8:
                input_11 = [temp_var for temp_var in np.average(temp[0],axis=0)]
                input_12 = [temp_var for temp_var in np.std(temp[0],axis=0)]
                input_21 = [temp_var for temp_var in np.average(temp[5],axis=0)]
                input_22 = [temp_var for temp_var in np.std(temp[5],axis=0)]
                input_31 = [temp_var for temp_var in np.average(temp[6],axis=0)]
                input_32 = [temp_var for temp_var in np.std(temp[6],axis=0)]
                files = input_11, input_12, input_21, input_22, input_31, input_32
                with open(f'/rq2_{method_setting}_{size}','wb') as pklfile:
                    pickle.dump(files, pklfile)

            for temp1 in range(7):
                result[idx,:,temp1,i] = temp[temp1].flatten()

    result2 = np.zeros((100,2))
    for i,size in enumerate([300,500,700]):
        for temp1 in range(7):
            for temp3 in range(2):
                result2[:,temp3] = result[temp3,:,temp1,i]

            d = result2[:,1] - result2[:,0]
            res = wilcoxon(d)
            print(method_setting,temp1,size,res.pvalue<0.05,(res.pvalue<=0.05)and(res.pvalue>0.001),res.pvalue<=0.001)
            print('\n')
            df = pd.DataFrame(result2)
            df.to_csv(f"test_{temp1}_size_{size}.csv",index=False,encoding='utf-8-sig')
    with open(f'/rq2_{method_setting}_{size}','wb') as pklfile:
        pickle.dump(files, pklfile)


if table == 3 or fig == 9 or fig == 10 or fig == 11:
    for model_name in ['codebert','roberta','rta']:
    # for model_name in ['roberta','rta']:
        print(model_name)
        result = np.zeros((2,100,7,3))
        for idx, method_setting in enumerate(['random','normalized_sum']):
            f1_flag, f2_flag, f3_flag, f4_flag, f5_flag, MODE = get_flags()
            for i,size in enumerate([300,500,700]):
                temp = np.zeros((7,10,10))
                for run in range(1, sample_times+1):
                    for temp_step in range(1, step+1):
                        if method_setting == 'normalized_sum':
                            with open(f'/rq3/data/rq3_initial_size{size}_sample_size{size}_run_{run}_step{temp_step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_model{model_name}.pkl','rb') as pklfile:
                                files = pickle.load(pklfile)
                            _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files

                        else:
                            with open(f'/rq3/data/rq3_initial_size{size}_sample_size{size}_run_{run}_step{temp_step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_model{model_name}.pkl','rb') as pklfile:
                                files = pickle.load(pklfile)
                            _, _, _, metrics_update, saved_indices, _, read_values, iden_values  = files
                        temp[5][run-1][temp_step-1] = np.average(read_values)
                        temp[6][run-1][temp_step-1] = np.average(iden_values)


                    if method_setting == 'normalized_sum':
                        with open(f'/rq3/data/rq3_initial_size{size}_sample_size{size}_run_{run}_step{step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_model{model_name}.pkl','rb') as pklfile:
                            files = pickle.load(pklfile)
                        _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files
                    else:
                        with open(f'/rq3/data/rq3_initial_size{size}_sample_size{size}_run_{run}_step{step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_model{model_name}.pkl','rb') as pklfile:
                            files = pickle.load(pklfile)
                        _, _, _, metrics_update, saved_indices, _, read_values, iden_values  = files
                    for temp_step in range(1,11):
                        for j in range(5):
                            temp[j][run-1][temp_step-1] += metrics_update[j][temp_step]

                print(f'{model_name}， {method_setting}, query size {size}')
                
                # table 3
                if table == 3:
                    print(f'&&{round(np.average(temp[0]),3)}&&{round(np.average(temp[1]),3)}&&{round(np.average(temp[3]),3)}&&{round(np.average(temp[4]),3)}&&{round(np.average(temp[5]),3)}&&{round(np.average(temp[6]),3)}\\\\')
                    print(f'&&{round(np.std(temp[0]),3)}&&{round(np.std(temp[1]),3)}&&{round(np.std(temp[3]),3)}&&{round(np.std(temp[4]),3)}&&{round(np.std(temp[5]),3)}&&{round(np.std(temp[6]),3)}\\\\')
                    
                # fig 9,10,11
                if fig == 9 or fig == 10 or fig == 11:
                    input_11 = [temp_var for temp_var in np.average(temp[0],axis=0)]
                    input_12 = [temp_var for temp_var in np.std(temp[0],axis=0)]
                    input_21 = [temp_var for temp_var in np.average(temp[5],axis=0)]
                    input_22 = [temp_var for temp_var in np.std(temp[5],axis=0)]
                    input_31 = [temp_var for temp_var in np.average(temp[6],axis=0)]
                    input_32 = [temp_var for temp_var in np.std(temp[6],axis=0)]
                    files = input_11, input_12, input_21, input_22, input_31, input_32
                    with open(f'/rq3_{model_name}_{method_setting}_{size}','wb') as pklfile:
                        pickle.dump(files, pklfile)

                for temp1 in range(7):
                    result[idx,:,temp1,i] = temp[temp1].flatten()

        result2 = np.zeros((100,2))
        for i,size in enumerate([300,500,700]):
            for temp1 in range(7):
                for temp3 in range(2):
                    result2[:,temp3] = result[temp3,:,temp1,i]

                d = result2[:,1] - result2[:,0]
                res = wilcoxon(d)
                print(method_setting,temp1,size,res.pvalue<0.05,(res.pvalue<=0.05)and(res.pvalue>0.001),res.pvalue<=0.001)
            print('\n')

if table == 5 or fig == 12:
    method_setting = 'normalized_sum'
    result = np.zeros((5,100,7,3))
    f1_flag, f2_flag, f3_flag, f4_flag, f5_flag, MODE = get_flags()
    for idx,sota in enumerate(['ours','Wu2021','Ge2021','Fang2015','Tu2022']):
    # for model_name in ['roberta','rta']:
        print(sota)
        for i,size in enumerate([300,500,700]):

            temp = np.zeros((7,10,10))
            for run in range(1, sample_times+1):
                for temp_step in range(1, step+1):
                    if sota == 'ours':
                        with open(f'/rq2/data/rq2_initial_size{size}_sample_size{size}_run_{run}_step{temp_step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_ratio1.pkl','rb') as pklfile:
                            files = pickle.load(pklfile)
                        _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files

                    else:
                        with open(f'/rq4/data/rq4_initial_size{size}_sample_size{size}_run_{run}_step{temp_step}-{sample_times}_sota_{sota}.pkl','rb') as pklfile:
                            files = pickle.load(pklfile)
                        _, _, _, metrics_update, _, _, read_values, iden_values  = files
            
                    temp[5][run-1][temp_step-1] = np.average(read_values)
                    temp[6][run-1][temp_step-1] = np.average(iden_values)

                if sota == 'ours':
                    with open(f'/rq2/data/rq2_initial_size{size}_sample_size{size}_run_{run}_step{step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_ratio1.pkl','rb') as pklfile:
                        files = pickle.load(pklfile)
                    _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files

                else:
                    with open(f'/rq4/data/rq4_initial_size{size}_sample_size{size}_run_{run}_step{step}-{sample_times}_sota_{sota}.pkl','rb') as pklfile:
                        files = pickle.load(pklfile)
                    _, _, _, metrics_update, _, _, read_values, iden_values  = files
                for temp_step in range(1,11):
                    for j in range(5):
                        temp[j][run-1][temp_step-1] += metrics_update[j][temp_step]

            # table 5
            if table == 5:
                print(f'{sota}， {method_setting}, query size {size}')
                print(f'&&{round(np.average(temp[0]),3)}&&{round(np.average(temp[1]),3)}&&{round(np.average(temp[3]),3)}&&{round(np.average(temp[4]),3)}&&{round(np.average(temp[5]),3)}&&{round(np.average(temp[6]),3)}\\\\')
                print(f'&&{round(np.std(temp[0]),3)}&&{round(np.std(temp[1]),3)}&&{round(np.std(temp[3]),3)}&&{round(np.std(temp[4]),3)}&&{round(np.std(temp[5]),3)}&&{round(np.std(temp[6]),3)}\\\\')
                
            # fig 12
            if fig == 12:
                input_11 = [temp_var for temp_var in np.average(temp[0],axis=0)]
                input_12 = [temp_var for temp_var in np.std(temp[0],axis=0)]
                input_21 = [temp_var for temp_var in np.average(temp[5],axis=0)]
                input_22 = [temp_var for temp_var in np.std(temp[5],axis=0)]
                input_31 = [temp_var for temp_var in np.average(temp[6],axis=0)]
                input_32 = [temp_var for temp_var in np.std(temp[6],axis=0)]
                files = input_11, input_12, input_21, input_22, input_31, input_32
                with open(f'/rq4_{sota}_{size}','wb') as pklfile:
                    pickle.dump(files, pklfile)


if table == 6 or fig == 13:
    read_num_random = 0
    read_num_ours = 0
    iden_num_random = 0
    iden_num_ours = 0
    time_num_random = 0
    time_num_ours = 0
    counter_random = 0 
    counter_ours = 0 

    list_temp = range(1,11)

    for i in list_temp:

        with open(f'/participant/id2label_{i}.pkl','rb') as pklfile:
            id2label_dict, temp = pickle.load(pklfile)

        # labels = temp['id'].apply(lambda x: id2label_dict[x])

        df = pd.read_csv(f"/rq5/result_{i}.csv",header=None)
        df.columns = ['temp_id','annoation','read','iden','time']
        df['label'] = temp['id'].apply(lambda x: id2label_dict[x])

        for index, row in df.iterrows():
            if row['label'] == 1:
                counter_ours += 1
                read_num_ours += row['read']
                iden_num_ours += row['iden']
                time_num_ours += row['time']
            else:
                counter_random += 1
                read_num_random += row['read']
                iden_num_random += row['iden'] 
                time_num_random += row['time']


    # table 6
    print('counter_random',counter_random)
    print('Random read is ',read_num_random/300)
    print('Random iden is ',iden_num_random/300)
    print('Random time is ',time_num_random/300)
    print('Random total time is ',time_num_random)
    print('Random annotated per hour is ',60*60/(time_num_random/300))
    print('Random money per issue is ',10.24/(60*60/(time_num_random/300)))
    print('Random money per 300 issue is ',10.24/(60*60/(time_num_random/300))*300)
    print('Random money per 3000 issue is ',10.24/(60*60/(time_num_random/300))*3000)


    print('counter_ours',counter_ours)
    print('Ours read is ',read_num_ours/300)
    print('Ours iden is ',iden_num_ours/300)
    print('Ours time is ',time_num_ours/300)
    print('Ours total time is ',time_num_ours)
    print('Ours annotated per hour is ',60*60/(time_num_ours/300))
    print('Ours money per issue is ',10.24/(60*60/(time_num_ours/300)))
    print('Ours money per 300 issue is ',10.24/(60*60/(time_num_ours/300))*300)
    print('Ours money per 3000 issue is ',10.24/(60*60/(time_num_ours/300))*3000)

    def a12effect(group1,group2):
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std

        print("Cohen's d:", cohens_d)


    list_temp = range(1,11)
    temp1_read = []
    temp2_read = []
    temp1_iden = []
    temp2_iden = []
    temp1_time = []
    temp2_time = []
    for i in list_temp:

        read_num_random = []
        read_num_ours = []
        iden_num_random = []
        iden_num_ours = []
        time_num_random = []
        time_num_ours = []

        with open(f'/participant/id2label_{i}.pkl','rb') as pklfile:
            id2label_dict, temp = pickle.load(pklfile)

        # labels = temp['id'].apply(lambda x: id2label_dict[x])

        df = pd.read_csv(f"/rq5/result_{i}.csv",header=None)
        df.columns = ['temp_id','annoation','read','iden','time']
        df['label'] = temp['id'].apply(lambda x: id2label_dict[x])

        for index, row in df.iterrows():
            if row['label'] == 1:
                counter_ours += 1
                read_num_ours.append(row['read'])
                iden_num_ours.append(row['iden'])
                time_num_ours.append(row['time'])
            else:
                counter_random += 1
                read_num_random.append(row['read'])
                iden_num_random.append(row['iden'])
                time_num_random.append(row['time'])
                # print('row[read]',row['read'])
        
        # fig 13
        print(f"participant {i}")
        print(f'read random {read_num_random/30}')
        print(f'read ours ',{read_num_ours/30})
        print('iden random ',iden_num_random/30)
        print('iden ours ',iden_num_ours/30)
        print('time random ', round(time_num_random/30*300/60/60 , 2))
        print('time ours ', round(time_num_ours/30*300/60/60, 2))
        temp1_read.append(read_num_random/30)
        temp2_read.append(read_num_ours/30)
        temp1_iden.append(iden_num_random/30)
        temp2_iden.append(iden_num_ours/30)
        temp1_time.append(time_num_random/30)
        temp2_time.append(time_num_ours/30)

        print(temp['id'])
    a12effect(temp1_read,temp2_read)
    a12effect(temp1_iden,temp2_iden)
    a12effect(temp1_time,temp2_time)


# fig 14
if fig == 14:
    with open(f'/initial_data/run{3}_sample_size{300}_initial_data.pkl','rb') as pklfile:
        files = pickle.load(pklfile)
    _, _, bug_pool_ids, _ = files

    norm_flag = False
    for step in range(10):

        method_setting = "normalized_sum"
        f1_flag, f2_flag, f3_flag, f4_flag, f5_flag, MODE = get_flags()

        if step == 0:
            with open(f'/rq9/data/rq9_mid_initial_size{300}_sample_size{300}_run_{3}_step{step+1}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo{2}_ratio{1}.pkl','rb') as pklfile:
                files = pickle.load(pklfile)
            bug_annotated_ids, bug_pool_ids_current, y_true_labeled, metrics_mixed, saved_indices, result, read_values, iden_values, tmp, bug_annotated_ids_real, y_true_labeled_real, cer, read, iden, bug_pool_ids_last, _ = files
        else:
            with open(f'/rq9/data/rq9_mid_initial_size{300}_sample_size{300}_run_{3}_step{step+1}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo{2}_ratio{1}.pkl','rb') as pklfile:
                files = pickle.load(pklfile)
            bug_annotated_ids, bug_pool_ids_current, y_true_labeled, metrics_mixed, saved_indices, result, read_values, iden_values, tmp, bug_annotated_ids_real, y_true_labeled_real, cer, read, iden, bug_pool_ids_last, _, cer_last, read_last, iden_last = files
        # converted_index = X_train.iloc[saved_indices].index

        costs = np.array([cer, -read, -iden]).T
        scaler = MinMaxScaler()
        scaler.fit(costs)
        normalized_costs = scaler.transform(costs)

        pareto_indices = pareto_front(normalized_costs)

        temp = normalized_costs.T
        obj = temp[0]+temp[1]+temp[2]
        
        indices = np.argpartition(obj,300)[:300]
        # tenk_indices = np.argpartition(obj,300)[:300]
        tenk_indices = np.argpartition(obj,1000)[:1000]
        result = obj[indices]
        print(step)

        if norm_flag:
            x_lower, x_upper = 0, 1
            y_lower, y_upper = 0, 1
            z_lower, z_upper = 0, 1
        else:
            x_lower, x_upper = 0, 0.5
            y_lower, y_upper = -1000, 207
            z_lower, z_upper = 0, 1

        if step == 0:
            filtered_indices = np.setdiff1d(tenk_indices, indices)
            if norm_flag:
                x1, y1, z1 = 1 - temp[0][filtered_indices], 1- temp[1][filtered_indices], 1-temp[2][filtered_indices]
                x2, y2, z2 = 1 - temp[0][indices], 1- temp[1][indices], 1-temp[2][indices]
                # x3, y3, z3 = 1 - temp[0][pareto_indices], 1- temp[1][pareto_indices], 1-temp[2][pareto_indices]

            else:
                x1, y1, z1 = 0.5-cer[filtered_indices], read[filtered_indices], iden[filtered_indices]
                x2, y2, z2 = 0.5-cer[indices], read[indices], iden[indices]
                # x3, y3, z3 = 0.5-cer[pareto_indices], read[pareto_indices], iden[pareto_indices]

            x1 = np.clip(x1, x_lower, x_upper)
            x2 = np.clip(x2, x_lower, x_upper)
            # x3 = np.clip(x3, x_lower, x_upper)

            y1 = np.clip(y1, y_lower, y_upper)
            y2 = np.clip(y2, y_lower, y_upper)
            # y3 = np.clip(y3, y_lower, y_upper)

            z1 = np.clip(z1, z_lower, z_upper)
            z2 = np.clip(z2, z_lower, z_upper)
            # z3 = np.clip(z3, z_lower, z_upper)

            data = np.array([x1, y1, z1])
            print('len(x1)',len(x1))
            with open(f'test_step{step+1}_1.dat', 'w') as datfile:
                np.savetxt(datfile, data.T)
            data = np.array([x2, y2, z2])
            with open(f'test_step{step+1}_2.dat', 'w') as datfile:
                np.savetxt(datfile, data.T)

            # last_outer_top300_indices = bug_pool_ids[indices]
            last_outer_filtered_indices = bug_pool_ids_last[filtered_indices]
            # print('bug_pool_ids',len(bug_pool_ids))
        

        else:
            filtered_indices = np.setdiff1d(tenk_indices, indices)
            last_indices = np.isin(bug_pool_ids_last, last_outer_filtered_indices)
            last_top300_indices = np.isin(bug_pool_ids_last, saved_indices)

            if norm_flag:
                x1, y1, z1 = 1 - temp[0][filtered_indices], 1- temp[1][filtered_indices], 1 - temp[2][filtered_indices]
                x2, y2, z2 = 1 - temp[0][indices], 1- temp[1][indices], 1 - temp[2][indices]
                x4, y4, z4 = 1 - temp[0][last_indices], 1- temp[1][last_indices], 1 - temp[2][last_indices]
                # x3, y3, z3 = 1 - temp[0][pareto_indices], 1- temp[1][pareto_indices], 1 - temp[2][pareto_indices]
            else:
                x1, y1, z1 = 0.5-cer[filtered_indices], read[filtered_indices], iden[filtered_indices]
                print('len(x1)',len(x1))
                x2, y2, z2 = 0.5-cer[indices], read[indices], iden[indices]
                print('len(x2)',len(x2))
                x4, y4, z4 = 0.5-cer[last_indices], read[last_indices], iden[last_indices]
                # print('len(x3)',len(x3))
                # x3, y3, z3 = 0.5-cer[pareto_indices], read[pareto_indices], iden[pareto_indices]
                print('len(x4)',len(x4))
                # x5, y5, z5 = 0.5-cer_last, read_last, iden_last

            x1 = np.clip(x1, x_lower, x_upper)
            x2 = np.clip(x2, x_lower, x_upper)
            # x3 = np.clip(x3, x_lower, x_upper)
            x4 = np.clip(x4, x_lower, x_upper)

            y1 = np.clip(y1, y_lower, y_upper)
            y2 = np.clip(y2, y_lower, y_upper)
            # y3 = np.clip(y3, y_lower, y_upper)
            y4 = np.clip(y4, y_lower, y_upper)

            z1 = np.clip(z1, z_lower, z_upper)
            z2 = np.clip(z2, z_lower, z_upper)
            # z3 = np.clip(z3, z_lower, z_upper)
            z4 = np.clip(z4, z_lower, z_upper)


            data = np.array([x1, y1, z1])
            with open(f'test_step{step+1}_1.dat', 'a') as datfile:
                np.savetxt(datfile, data.T)
            data = np.array([x2, y2, z2])
            with open(f'test_step{step+1}_2.dat', 'a') as datfile:
                np.savetxt(datfile, data.T)
            # data = np.array([x3, y3, z3])
            # with open(f'test_step{step+1}_3.dat', 'a') as datfile:
            #     np.savetxt(datfile, data.T)
            data = np.array([x4, y4, z4])
            with open(f'test_step{step+1}_4.dat', 'a') as datfile:
                np.savetxt(datfile, data.T)

            last_outer_top300_indices = bug_pool_ids_last[indices]
            last_outer_filtered_indices = bug_pool_ids_last[filtered_indices]


# fig 15
if fig == 15:
    result = np.zeros((3,100,7,3))
    method_setting = 'normalized_sum'
    f1_flag, f2_flag, f3_flag, f4_flag, f5_flag, MODE = get_flags()
    for idx, ratio in enumerate([1,2]):

        if ratio == 1:
            iter_size = [300,500,700]
        else:
            iter_size = [300]   
        for size in iter_size:
            if size == 300:
                i = 0
            elif size == 500:
                i = 1
            else:
                i = 2
                print('yes')
            result2 = np.zeros((sample_times,3))
            temp = np.zeros((7,10,10))
            for run in range(1, sample_times+1):
                for temp_step in range(1, step+1):
                    with open(f'/rq2/data/rq2_initial_size{size}_sample_size{size}_run_{run}_step{temp_step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_ratio{ratio}.pkl','rb') as pklfile:
                        files = pickle.load(pklfile)
                    _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files

                    temp[5][run-1][temp_step-1] = np.average(read_values)
                    temp[6][run-1][temp_step-1] = np.average(iden_values)

                with open(f'/rq2/data/rq2_initial_size{size}_sample_size{size}_run_{run}_step{step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_ratio{ratio}.pkl','rb') as pklfile:
                    files = pickle.load(pklfile)
                _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files

                for temp_step in range(1,11):
                    for j in range(5):
                        temp[j][run-1][temp_step-1] += metrics_update[j][temp_step]

            print(f'{ratio}, query size {size}')
            # print(f'&&{round(np.average(temp[0]),3)} (\$\\pm {round(np.std(temp[0]),3)}\$)&&{round(np.average(temp[1]),3)}&&{round(np.average(temp[3]),3)}&&{round(np.average(temp[4]),3)}&&{round(np.average(temp[5]),3)}&&{round(np.average(temp[6]),3)}\\\\')

            print(f'&&{round(np.average(temp[0]),3)} ($\\pm {round(np.std(temp[0]),3)}$)&&{round(np.average(temp[1]),3)} ($\\pm {round(np.std(temp[1]),3)}$)&&{round(np.average(temp[3]),3)} ($\\pm {round(np.std(temp[3]),3)}$)&&{round(np.average(temp[4]),3)} ($\\pm {round(np.std(temp[4]),3)}$)&&{round(np.average(temp[5]),3)} ($\\pm {round(np.std(temp[5]),3)}$)&&{round(np.average(temp[6]),3)} ($\\pm {round(np.std(temp[6]),3)}$)')
            # print(f'&&{round(np.std(temp[0]),3)}&&{round(np.std(temp[1]),3)}&&{round(np.std(temp[3]),3)}&&{round(np.std(temp[4]),3)}&&{round(np.std(temp[5]),3)}&&{round(np.std(temp[6]),3)}\\\\')
    
            for temp1 in range(7):
                result[idx,:,temp1,i] = temp[temp1].flatten()

    method_setting = 'normalized_sum'
    f1_flag, f2_flag, f3_flag, f4_flag, f5_flag, MODE = get_flags()
    for idx, ratio in enumerate([2,3]):
        if ratio == 2:
            idx = 1
            iter_size = [500,700]
        else:
            idx = 2
            iter_size = [300,500,700]   
        for size in iter_size:
            if size == 300:
                i = 0
            elif size == 500:
                i = 1
            else:
                i = 2
            result2 = np.zeros((sample_times,3))
            temp = np.zeros((7,10,10))
            for run in range(1, sample_times+1):
                for temp_step in range(1, step+1):
                    with open(f'/rq7/data/rq7_initial_size{size}_sample_size{size}_run_{run}_step{temp_step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_ratio{ratio}.pkl','rb') as pklfile:
                        files = pickle.load(pklfile)
                    if len(files) ==  11:
                        _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files
                    else:
                        _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values, _ = files


                    temp[5][run-1][temp_step-1] = np.average(read_values)
                    temp[6][run-1][temp_step-1] = np.average(iden_values)
                with open(f'/rq7/data/rq7_initial_size{size}_sample_size{size}_run_{run}_step{step}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo2_ratio{ratio}.pkl','rb') as pklfile:
                    files = pickle.load(pklfile)
                if len(files) ==  11:
                    _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values  = files
                else:
                    _, _, _, _, metrics_update, _, _, _, _, read_values, iden_values, _ = files


                for temp_step in range(1,11):
                    for j in range(5):
                        temp[j][run-1][temp_step-1] += metrics_update[j][temp_step]

            print(f'{ratio}, query size {size}')
            # print(f'&&{round(np.average(temp[0]),3)}&&{round(np.average(temp[1]),3)}&&{round(np.average(temp[3]),3)}&&{round(np.average(temp[4]),3)}&&{round(np.average(temp[5]),3)}&&{round(np.average(temp[6]),3)}\\\\')
            # print(f'&&{round(np.std(temp[0]),3)}&&{round(np.std(temp[1]),3)}&&{round(np.std(temp[3]),3)}&&{round(np.std(temp[4]),3)}&&{round(np.std(temp[5]),3)}&&{round(np.std(temp[6]),3)}\\\\')
            print(f'&&{round(np.average(temp[0]),3)} ($\\pm {round(np.std(temp[0]),3)}$)&&{round(np.average(temp[1]),3)} ($\\pm {round(np.std(temp[1]),3)}$)&&{round(np.average(temp[3]),3)} ($\\pm {round(np.std(temp[3]),3)}$)&&{round(np.average(temp[4]),3)} ($\\pm {round(np.std(temp[4]),3)}$)&&{round(np.average(temp[5]),3)} ($\\pm {round(np.std(temp[5]),3)}$)&&{round(np.average(temp[6]),3)} ($\\pm {round(np.std(temp[6]),3)}$)')

            for temp1 in range(7):
                result[idx,:,temp1,i] = temp[temp1].flatten()
                # print(result[idx,:,temp1,i])



    result2 = np.zeros((100,3))
    for i,size in enumerate([300,500,700]):
        for temp1 in range(7):
            for temp3 in range(3):
                result2[:,temp3] = result[temp3,:,temp1,i]
                # print(result[temp3,:,temp1,i])
            df = pd.DataFrame(result2)
            df.to_csv(f"test_{temp1}_size_{size}.csv",index=False,encoding='utf-8-sig')

