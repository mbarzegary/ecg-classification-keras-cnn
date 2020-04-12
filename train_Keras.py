from load_MITBIH import load_mit_db

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

import os
import time
import settings

import matplotlib.pyplot as plt
import matplotlib


def create_svm_model_name(model_path, winL, winR, do_preprocess, 
    maxRR, use_RR, norm_RR, compute_morph,  
    leads_flag, reduced_DS, delimiter):

    if reduced_DS == True:
        model_path = model_path + delimiter + 'exp_2'

    if leads_flag[0] == 1:
        model_path = model_path + delimiter + 'MLII'
    
    if leads_flag[1] == 1:
        model_path = model_path + delimiter + 'V1'

    if do_preprocess:
        model_path = model_path + delimiter + 'rm_bsln'

    if maxRR:
        model_path = model_path + delimiter + 'maxRR'

    if use_RR:
        model_path = model_path + delimiter + 'RR'
    
    if norm_RR:
        model_path = model_path + delimiter + 'norm_RR'
    
    for descp in compute_morph:
        model_path = model_path + delimiter + descp

    return model_path

def main_DP(winL=90, winR=90, do_preprocess=True, 
    maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, reduced_DS = False, leads_flag = [1,0]):
    print("Runing train_Keras.py for Differential Privacy!")

    db_path = settings.db_path
    
    # Load train data 
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
        maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag)

    # Load test data
    [eval_features, eval_labels, eval_patient_num_beats] = load_mit_db('DS2', winL, winR, do_preprocess, 
        maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag)

    scaler = StandardScaler()
    scaler.fit(tr_features)
    tr_features_scaled = scaler.transform(tr_features)
    eval_features_scaled = scaler.transform(eval_features)

    model_path = db_path + 'keras_models/'

    model_path = create_svm_model_name(model_path, winL, winR, do_preprocess,
        maxRR, use_RR, norm_RR, compute_morph,
        leads_flag, reduced_DS, '_')

    model_path = model_path + '.h5'

    print(("Training model on MIT-BIH DS1: " + model_path + "..."))

    if 1==2:#os.path.isfile(model_svm_path):
        # Load the trained model!
        mlp_model = load_model(model_path)

    else:
        # print(tr_features_scaled.shape[1])

        l2_norm_clip = 1.5
        noise_multiplier = 1.4
        num_microbatches = 250
        learning_rate = 0.25

        mlp_model = Sequential()
        mlp_model.add(Dense(100, input_dim=tr_features_scaled.shape[1], activation='relu'))
        # mlp_model.add(Dropout(0.5))
        # mlp_model.add(Dense(64, activation='relu'))
        mlp_model.add(Dropout(0.5))
        mlp_model.add(Dense(1, activation='sigmoid'))

        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)

        mlp_model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

        # Let's Train!
        start = time.time()
        mlp_model.fit(tr_features_scaled, tr_labels, epochs = 5, batch_size = 128)
        end = time.time()

        print(("Trained completed!\n\t" + model_path + "\n \
            \tTime required: " + str(format(end - start, '.2f')) + " sec" ))

        # Save trained MLP model
        mlp_model.save(model_path)

    # Test the model
    print(("Testing model on MIT-BIH DS2: " + model_path + "..."))


    # Evaluate the model with new data
    predictions = mlp_model.predict(eval_features_scaled)
    predictions = (predictions.squeeze() > 0.5)
    print(confusion_matrix(eval_labels, predictions))
    print(classification_report(eval_labels, predictions))
    print("Accuracy: {0}".format(accuracy_score(eval_labels, predictions)))

    compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=tr_features_scaled.shape[0], batch_size=128, noise_multiplier=noise_multiplier, epochs=5, delta=1e-5)


    

def main(winL=90, winR=90, do_preprocess=True, 
    maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, reduced_DS = False, leads_flag = [1,0]):
    print("Runing train_Keras.py!")

    db_path = settings.db_path
    
    # Load train data 
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
        maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag)

    # Load test data
    [eval_features, eval_labels, eval_patient_num_beats] = load_mit_db('DS2', winL, winR, do_preprocess, 
        maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag)

    scaler = StandardScaler()
    scaler.fit(tr_features)
    tr_features_scaled = scaler.transform(tr_features)
    eval_features_scaled = scaler.transform(eval_features)

    model_path = db_path + 'keras_models/'

    model_path = create_svm_model_name(model_path, winL, winR, do_preprocess,
        maxRR, use_RR, norm_RR, compute_morph,
        leads_flag, reduced_DS, '_')

    model_path = model_path + '.h5'

    print(("Training model on MIT-BIH DS1: " + model_path + "..."))

    if 1==2:#os.path.isfile(model_svm_path):
        # Load the trained model!
        mlp_model = load_model(model_path)

    else:
        # print(tr_features_scaled.shape[1])

        mlp_model = Sequential()
        mlp_model.add(Dense(100, input_dim=tr_features_scaled.shape[1], activation='relu'))
        mlp_model.add(Dropout(0.5))
        # mlp_model.add(Dense(64, activation='relu'))
        # mlp_model.add(Dropout(0.5))
        mlp_model.add(Dense(1, activation='sigmoid'))

        mlp_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
        
        # Let's Train!
        start = time.time()
        mlp_model.fit(tr_features_scaled, tr_labels, epochs = 5, batch_size = 128)
        end = time.time()

        print(("Trained completed!\n\t" + model_path + "\n \
            \tTime required: " + str(format(end - start, '.2f')) + " sec" ))

        # Save trained MLP model
        mlp_model.save(model_path)

    # Test the model
    print(("Testing model on MIT-BIH DS2: " + model_path + "..."))


    # Evaluate the model with new data
    predictions = mlp_model.predict(eval_features_scaled)
    predictions = (predictions.squeeze() > 0.5)
    print(confusion_matrix(eval_labels, predictions))
    print(classification_report(eval_labels, predictions))
    print("Accuracy: {0}".format(accuracy_score(eval_labels, predictions)))


    # print("Soft max:")
    # pred_softmax = mlp_model.predict(eval_features_scaled)
    # print(pred_softmax)

    # comp = np.zeros([predictions.size, 4]) # Columns: correct_0, correct_1, incorrect_0, incorrect_1
    # for i in range(predictions.size):
    #     # if eval_labels[i] == 1:
    #     #     print("i = {0}, label={1}".format(i, eval_labels[i]))
    #     # if predictions[i] != eval_labels[i]:
    #     #     print("Prediction:{0}, Actual:{1}, Softmax:{2}".format(predictions[i], eval_labels[i], pred_softmax[i,:]))
    #     if predictions[i] == eval_labels[i]: # Correct
    #         if predictions[i] == 0:
    #             comp[i,0] = pred_softmax[i,0]
    #         else:
    #             comp[i,1] = pred_softmax[i,1]
    #     else: # Incorrect
    #         if predictions[i] == 0:
    #             comp[i,2] = pred_softmax[i,0]
    #         else:
    #             comp[i,3] = pred_softmax[i,1]
    
    # comp[comp == 0] = np.nan
    # print(np.nanmean(comp, axis=0))

    # hist_correct0 = [pred_softmax[i,0] for i in range(predictions.size) if predictions[i] == 0 and predictions[i] == eval_labels[i]]
    # hist_correct1 = [pred_softmax[i,1] for i in range(predictions.size) if predictions[i] == 1 and predictions[i] == eval_labels[i]]
    # hist_incorrect0 = [pred_softmax[i,0] for i in range(predictions.size) if predictions[i] == 0 and predictions[i] != eval_labels[i]]
    # hist_incorrect1 = [pred_softmax[i,1] for i in range(predictions.size) if predictions[i] == 1 and predictions[i] != eval_labels[i]]

    # font = {'weight' : 'bold',
    #         'size'   : 16}

    # matplotlib.rc('font', **font)

    # plt.hist(hist_correct0, bins=50)
    # plt.xlabel("Softmax probability")
    # plt.ylabel("Number of samples")
    # plt.title("Correctly predicted Normal")
    # plt.tight_layout()
    
    # plt.figure()
    # plt.hist(hist_correct1, bins=50)
    # plt.xlabel("Softmax probability")
    # plt.ylabel("Number of samples")
    # plt.title("Correctly predicted Abnormal")
    # plt.tight_layout()

    # plt.figure()
    # plt.hist(hist_incorrect0, bins=50)
    # plt.xlabel("Softmax probability")
    # plt.ylabel("Number of samples")
    # plt.title("Incorrectly predicted Normal")
    # plt.tight_layout()

    # plt.figure()
    # plt.hist(hist_incorrect1, bins=50)
    # plt.xlabel("Softmax probability")
    # plt.ylabel("Number of samples")
    # plt.title("Incorrectly predicted Abnormal")
    # plt.tight_layout()

    # plt.show()