from collections import defaultdict
import numpy as np
import os
import matplotlib as mpl
mpl.use('pdf') 
import matplotlib.pyplot as plt






if __name__ == "__main__":



    fs = 10
    
  # experiment in which the number of frames is fixed to 120, and the number of classes varies
   
    num_classes         =        [1 ,          2,          3,         4,          5,            6,          7,       8]
    match_rate_ours     =        [.88,        .9,       .906,      .8191,       .94,        .9583,      .8657,      .8487]
    match_rate_baseline =        [.83,       .83,      .6850,     .6850,        .6762,     .6287,      .5525,     .5163]    


    plt.figure(dpi=250)
    plt.plot(num_classes, match_rate_ours, marker='o', label='RTVA-Multi')
    plt.plot(num_classes, match_rate_baseline, marker='x', label='ACTOR-Multi')
    #plt.xlim(60,220)
    plt.xlabel('classes', fontsize=fs)
    plt.ylabel('matching rate', fontsize=fs)
    plt.title('Semantic match rate vs number of classes for seq. legnth 120')
    plt.legend(loc='lower left')
    plt.savefig('match_rate_varynumclasses.png')


    #number of classes is 4, varying the sequence length
    seq_len =            [140,        180,       220,       260,       300,      340,      380,       420,        460,       500]
    match_rate_ours =    [0.7105,    .8340,     .8071,     .8054,    .8814,     .8809,   .8593,     .9063,      .9577,      .9456]
    match_rate_baseline =[.4123,     .3925,     .4357,     .4597,    .4872,     .5078,   .5260,     .5831,       .6133,      .6375]
    



    plt.figure(dpi=250)
    plt.plot(seq_len, match_rate_ours, marker='o', label='RTVA-Multi')
    plt.plot(seq_len, match_rate_baseline, marker='x', label='ACTOR-Multi')
    # plt.xlim(60,220)
    plt.xlabel('seq. length', fontsize=fs)
    plt.ylabel('matching rate', fontsize=fs)
    plt.title('Semantic match rate vs seq. length for 4 actions')
    plt.legend(loc='lower right')
    plt.savefig('match_rate_varyseqlength.png')




       #number of classes is 1, varying the sequence length
    seq_len             =   [20,       40,     60,      80,      100,         120,           140]
    match_rate_ours     =   [.5700,  .6300,   .7900,  .8300,    .8500,       .8700,         .8800]
    match_rate_baseline =   [.5100,  .6700,   .7500,  .8300,    .8300,       .8300,         .8600]
    



    plt.figure(dpi=250)
    plt.plot(seq_len, match_rate_ours, marker='o', label='RTVA-Single')
    plt.plot(seq_len, match_rate_baseline, marker='x', label='ACTOR')
    # plt.xlim(60,220)
    plt.xlabel('seq. length', fontsize=fs)
    plt.ylabel('matching rate', fontsize=fs)
    plt.title('Semantic match rate vs seq. length for 1 action')
    plt.legend(loc='lower right')
    plt.savefig('match_rate_1action_varyseqlength.png')


    
