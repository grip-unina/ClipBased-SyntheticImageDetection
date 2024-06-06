import os
import pandas
import numpy as np
from sklearn import metrics 


dict_metrics = {
    'auc' : lambda label, score: metrics.roc_auc_score(label,  score),
    'acc' : lambda label, score: metrics.balanced_accuracy_score(label, score>0),
}


def compute_metrics(input_csv, output_csv, metrics_fun):
    table = pandas.read_csv(output_csv)
    list_algs = [_ for _ in table.columns if _!='filename']
    table = pandas.read_csv(input_csv).merge(table, on=['filename', ])
    assert 'typ' in table
    list_typs = sorted([_ for _ in set(table['typ']) if _!='real'])
    table['label'] = table['typ']!='real'

    tab_metrics = pandas.DataFrame(index=list_algs, columns=list_typs)
    tab_metrics.loc[:, :] = np.nan
    for typ in list_typs:
        tab_typ = table[table['typ'].isin(['real', typ])]
        for alg in list_algs:    
            score = tab_typ[alg].values
            label = tab_typ['label'].values
            if np.all(np.isfinite(score))==False:
                continue
        
            tab_metrics.loc[alg, typ] = metrics_fun(label, score)
    tab_metrics.loc[:, 'AVG'] = tab_metrics.mean(1)
    
    return tab_metrics

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"  , '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv" , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--metrics" , '-w', type=str, help="type of metrics ('auc' or 'acc')", default="auc")
    parser.add_argument("--save_tab", '-t', type=str, help="The path of the metrics csv file", default=None)
    args = vars(parser.parse_args())
    
    tab_metrics = compute_metrics(args['in_csv'], args['out_csv'], dict_metrics[args['metrics']])
    tab_metrics.index.name = args['metrics']
    print(tab_metrics.to_string(float_format=lambda x: '%5.3f'%x))
    
    if args['save_tab'] is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args['save_tab'])), exist_ok=True)
        tab_metrics.to_csv(args['save_tab'])
    