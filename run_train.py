from train_keras import main, main_federated

# import tensorflow as tf
# # fixing cuDNN issue
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)


winL = 90
winR = 90
do_preprocess = True
maxRR = False
compute_morph = {''} # 'wvlt', 'HOS', 'myMorph', 'u-lbp'

reduced_DS = False # To select only patients in common with MLII and V1
leads_flag = [1,0] # MLII, V1

use_RR = False
norm_RR = False
compute_morph = {'raw'} 

# main(winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, reduced_DS, leads_flag)

# DP
# noise_list = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
# for noise in noise_list:
#     main(winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, reduced_DS, leads_flag, True, noise)

# Federated learning
main_federated(winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, reduced_DS, leads_flag)