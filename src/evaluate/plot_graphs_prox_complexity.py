from collections import defaultdict
import numpy as np
import os
import matplotlib as mpl
mpl.use('pdf') 
import matplotlib.pyplot as plt






if __name__ == "__main__":



    fs = 10
    
  # experiment in which the number of frames is fixed to 200, and the number of classes varies
   

    num_classes         =        [1 ,          2,          3,         4,          5,            6,          7]#,       8]
    match_rate_linear     =        [.7,        .839,       .8089,      .8219,     .806,         .7859,      .765]#,      .765] #todo full full linear
    match_rate_quad     =    [.66,       .861,      .8211,       .8321,     .806,         .7683,      .751]#,      .597] #todo full full full


    plt.figure(dpi=250)
    plt.plot(num_classes, match_rate_linear, marker='o', label='RTVA-Multi')
    plt.plot(num_classes, match_rate_quad, marker='x', label='RTVA-Multi-QUAD')
    #plt.xlim(1,220)
    plt.xlabel('classes', fontsize=fs)
    plt.ylabel('matching rate', fontsize=fs)
    plt.title('Semantic match rate vs number of classes for seq. legnth 200')
    plt.legend(loc='lower right')
    plt.savefig('match_rate_varynumclasses_complexity.png')

    # exps/recurrent_proxrefined_unlimited2/checkpoint_0700.pth.tar
    #number of classes is 8, varying the sequence length
    seq_len =            [120,       160,       200,       240,       280,       320,     360,      400]
    match_rate_linear =    [.833,    .86,        .765,       .735,      .745,     .7473,    .762,     .7286] # full full inear
    match_rate_quad =      [.880,    .813,       .751,       .748,      .763,      .711,     .709,     .6821] #full full full
    



    plt.figure(dpi=250)
    plt.plot(seq_len, match_rate_linear, marker='o', label='RTVA-Multi')
    plt.plot(seq_len, match_rate_quad, marker='x', label='RTVA-Multi-QUAD')
    # plt.xlim(60,220)
    plt.xlabel('seq. length', fontsize=fs)
    plt.ylabel('matching rate', fontsize=fs)
    plt.title('Semantic match rate vs seq. length for 8 actions')
    plt.legend(loc='lower left')
    plt.savefig('match_rate_varyseqlength_complexity.png')


    