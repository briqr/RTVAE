import os
import glob
import math
import numpy as np

from src.evaluate.tools import load_metrics


def get_gtname(mname):
    return mname + "_gt"


def get_genname(mname):
    return mname + "_gen"


def get_reconsname(mname):
    return mname + "_recons"


def valformat(val, power=3):
    p = float(pow(10, power))
    # "{:<04}".format(np.round(p*val).astype(int)/p)
    return str(np.round(p*val).astype(int)/p).ljust(4, "0")


def format_values(values, key, latex=True):
    mean = np.mean(values)

    if "accuracy" in key:
        mean = 100*mean
        values = 100*values
        smean = valformat(mean, 1)
    else:
        smean = valformat(mean, 2)

    #interval = valformat(1.96 * np.var(values), 2)  # [1:]
    interval = valformat(1.96 * np.std(values)/np.sqrt(len(values)), 2)  # [1:]

    if latex:
        string = rf"${smean}^{{\pm{interval}}}$"
    else:
        string = rf"{smean} +/- {interval}"
    return string


def print_results(folder, evaluation):
    evalpath = os.path.join(folder, evaluation)
    metrics = load_metrics(evalpath)

    a2m = metrics["feats"]

    if True or "fid_gen_test" in a2m: #todo
        keys = ["fid_{}_train", "fid_{}_test", "accuracy_{}_train",  "accuracy_{}_test", "diversity_{}_train", "multimodality_{}_train"]
    else:
        keys = ["fid_{}", "accuracy_{}", "diversity_{}", "multimodality_{}"]

    lines = ["gen", "recons", "gt"]
    # print the GT, only if it is computed with respect to "another" GT
    if "fid_gt2" in a2m:
        a2m["fid_gt"] = a2m["fid_gt2"]
        lines = ["gt"] + lines

    rows = []
    rows_latex = []

    for model in lines:
        row = ["{:6}".format(model)]
        row_latex = ["{:6}".format(model)]
        
        for key in keys:
            try:
                ckey = key.format(model)
                values = np.array([float(x) for x in a2m[ckey]])
                string_latex = format_values(values, key, latex=True)
                string = format_values(values, key, latex=False)
                row.append(ckey)
                row.append(': ,')
                row.append(string)
                row_latex.append(string_latex)
                #print('ckey: ', ckey)
            except KeyError:
                a = 0
                print('key error, %s'%ckey)
                continue
        rows.append(" | ".join(row))
        rows_latex.append(" & ".join(row_latex) + r"\\")

    table = "\n".join(rows)
    table_latex = "\n".join(rows_latex)
    print("Results")
    print(table)
    print()
    #print("Latex table")
    #print(table_latex)


if __name__ == "__main__":
    import argparse

    def parse_opts():
        parser = argparse.ArgumentParser()
        parser.add_argument("evalpath", help="name of the evaluation")
        return parser.parse_args()

    #opt = parse_opts()
    #evalpath = opt.evalpath
    epoch_range = [1, 2]
    
    for epoch in range(epoch_range[0], epoch_range[1], 20):
        #evalpath = 'exps/recurrent_humanact12_2/evaluation_metrics_%s_all.yaml'%str(epoch).zfill(4)
        evalpath = 'exps/recurrent_proxrefined_limit8/evaluation_metrics_0680_all.yaml'
        if not os.path.exists(evalpath):
            print('eval path %s does not exist' %evalpath)
            continue
        
        print('eval path: %s' %evalpath)
        folder, evaluation = os.path.split(evalpath)
        print_results(folder, evaluation)


# what i still don't understand is how the recurrent models with full attention is implemented as softmax?  I think it's ok.
# add relative encoding