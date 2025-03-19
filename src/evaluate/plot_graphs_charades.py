from collections import defaultdict
import numpy as np
import os
import matplotlib as mpl
mpl.use('pdf') 
import matplotlib.pyplot as plt






if __name__ == "__main__":



    fs = 10
    
  # experiment in which the number of frames is fixed to 400, and the number of classes varies, epoch 700 and 30 actions
   

   #subset of 30 actions
    num_classes         =        [1 ,          2,          3,         4,          5,            6,          7,       8]
    #match_rate_ours     =        [.48,        .651,      .73,         .713,     .725,         .684,      .695,      .737] #epoch 700
    match_rate_ours     =        [.54,        .689,      .72,         .713,     .7295,         .7293,      .691,      .765] #epoch 600
    match_rate_baseline     =    [.37,        .3544,      .4213,         .4215,     .483,         .481,      .451,      .597] #todo fill


    plt.figure(dpi=250)
    plt.plot(num_classes, match_rate_ours, marker='o', label='RTVA-Multi')
    plt.plot(num_classes, match_rate_baseline, marker='x', label='ACTOR-Multi')
    #plt.xlim(60,220)
    plt.xlabel('classes', fontsize=fs)
    plt.ylabel('matching rate', fontsize=fs)
    plt.title('Semantic match rate vs number of classes for seq. legnth 400')
    plt.legend(loc='lower right')
    plt.savefig('match_rate_varynumclasses_charades.png')


    #number of classes is 8, varying the sequence length
    seq_len =            [80,        120,       160,       200,       240,      280,      320,       360,        400]
    match_rate_ours =    [0.506,    .574,      .618,       .607,      .628,     .675,    .733,     .757,         .765] #epoch 600
    match_rate_baseline =[0.256,    .353,      .425,       .406,      .458,     .465,    .496,     .505,         .597] 
    



    plt.figure(dpi=250)
    plt.plot(seq_len, match_rate_ours, marker='o', label='RTVA-Multi')
    plt.plot(seq_len, match_rate_baseline, marker='x', label='ACTOR-Multi')
    # plt.xlim(60,220)
    plt.xlabel('seq. length', fontsize=fs)
    plt.ylabel('matching rate', fontsize=fs)
    plt.title('Semantic match rate vs seq. length for 8 actions')
    plt.legend(loc='lower right')
    plt.savefig('match_rate_varyseqlength_charades.png')



  # all actions
  
    num_classes         =        [1 ,          2,          3,         4,          5,            6,          7,       8] # fixed seq length of 400
    match_rate_ours     =        [.4,        .398,        .533,      .553,     .4768,         .4764,      .493,      .465] #epoch 550


    plt.figure(dpi=250)
    plt.plot(num_classes, match_rate_ours, marker='o', label='RTVA-Multi')
    #plt.xlim(60,220)
    plt.xlabel('classes', fontsize=fs)
    plt.ylabel('matching rate', fontsize=fs)
    plt.title('Semantic match rate vs number of classes for seq. legnth 400')
    plt.legend(loc='lower right')
    plt.savefig('match_rate_varynumclasses_charades_wholeset.png')


    #number of classes is 8, varying the sequence length
    seq_len =            [80,        120,       160,       200,       240,      280,      320,       360,        400]
    match_rate_ours =    [0.216,    .322,      .386,      .377,      .403,     .427,    .349,     .41,         .465] #epoch 550
    



    plt.figure(dpi=250)
    plt.plot(seq_len, match_rate_ours, marker='o', label='RTVA-Multi')
    # plt.xlim(60,220)
    plt.xlabel('seq. length', fontsize=fs)
    plt.ylabel('matching rate', fontsize=fs)
    plt.title('Semantic match rate vs seq. length for 8 actions')
    plt.legend(loc='lower right')
    plt.savefig('match_rate_varyseqlength_charades_wholeset.png')




    #    #number of classes is 1, varying the sequence length
    # seq_len             =   [20,       40,     60,      80,      100,         120,           140]
    # match_rate_ours     =   [.5700,  .6300,   .7900,  .8300,    .8500,       .8700,         .8800]
    # match_rate_baseline =   [.5100,  .6700,   .7500,  .8300,    .8300,       .8300,         .8600]
    



    # plt.figure(dpi=250)
    # plt.plot(seq_len, match_rate_ours, marker='o', label='RTVA-Single')
    # plt.plot(seq_len, match_rate_baseline, marker='x', label='ACTOR')
    # # plt.xlim(60,220)
    # plt.xlabel('seq. length', fontsize=fs)
    # plt.ylabel('matching rate', fontsize=fs)
    # plt.title('Semantic match rate vs seq. length for 1 action')
    # plt.legend(loc='lower right')
    # plt.savefig('match_rate_1action_varyseqlength.png')


    
