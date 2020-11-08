from load_MITBIH import load_mit_db

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_federated as tff

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

import glob
import os
import time
import tqdm
import imutils
import random
import h5py

import settings

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2

input_size = (64, 64)
change_probability = 0.3
rotate_range = 180
batch_size = 100
filters = (4, 4)
epochs = 20
regularizers = 0.0001
n_classes = 6
validation_split = 0.21
test_split = 0.15

# differential privacy parameters
l2_norm_clip = 1.5
# noise_multiplier = 1.4
num_microbatches = 100
learning_rate = 0.01

# federated learning parameters
NUM_CLIENTS = 10
NUM_EPOCHS = 1
BATCH_SIZE = 50
SHUFFLE_BUFFER = 50
PREFETCH_BUFFER = 25
NUM_ROUNDS = 20

random.seed()

def create_model_name(model_path, winL, winR, do_preprocess, 
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

def apply_cropping(filename, size):

    image = cv2.imread (filename, cv2.IMREAD_GRAYSCALE)

   # Left Top Crop
    crop = image[:48, :48]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-4] + '1' + '.png', crop)

    # Center Top Crop
    crop = image[:48, 8:56]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-4] + '2' + '.png', crop)

    # Right Top Crop
    crop = image[:48, 16:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-4] + '3' + '.png', crop)

    # Left Center Crop
    crop = image[8:56, :48]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-4] + '4' + '.png', crop)

    # Center Center Crop
    crop = image[8:56, 8:56]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-4] + '5' + '.png', crop)

    # Right Center Crop
    crop = image[8:56, 16:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-4] + '6' + '.png', crop)

    # Left Bottom Crop
    crop = image[8:, :48]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-4] + '7' + '.png', crop)

    # Center Bottom Crop
    crop = image[16:, 8:56]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-4] + '8' + '.png', crop)

    # Right Bottom Crop
    crop = image[16:, 16:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-4] + '9' + '.png', crop)


def get_images_data(features, labels, n_data, directory_name, augment=True):

    print(f"Creating image input files for {directory_name} ...")

    h5file = settings.images_hdf5_path + directory_name + ".h5"
    
    data_x = []
    data_y=[]
    augment_list = []

    if (os.path.isfile(h5file)):
        hf = h5py.File(h5file, 'r')
        data_x = np.array(hf.get('data_x'))
        data_y = np.array(hf.get('data_y'))
        augment_list = np.array(hf.get('augment_list'))
        hf.close()
    else:
            # for i in range (n_data):
        for i in tqdm.tqdm(range(n_data)):

            filename = f"./{directory_name}/{str(i)}-.png"

            if not os.path.isfile(filename):
                fig = plt.figure(figsize=(2, 2), dpi=32) # 64
                plt.plot(features[i], 'k')
                plt.xticks([]), plt.yticks([])
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                fig.savefig(filename, dpi=fig.dpi)
                plt.close(fig)
                if augment and labels[i] != 0: # do not augment for label 0
                    apply_cropping(filename, input_size)
                    augment_list.append(i)
                
                
            pattern = ""
            if augment and labels[i] != 0: # do not augment for label 0
                pattern = f"./{directory_name}/{str(i)}-*.png"
            else:
                pattern = f"./{directory_name}/{str(i)}-.png"

            listing = glob.glob(pattern)
            for file in listing:
                # im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # im_gray = cv2.resize(im_gray, (224, 224))
                # cv2.imwrite(filename, im_gray)
                image = cv2.imread (file, cv2.IMREAD_GRAYSCALE)
                # random rotate 
                if random.uniform(0, 1) < change_probability:
                    image = imutils.rotate(image, random.randint(-rotate_range, rotate_range))
                # random flip 
                if random.uniform(0, 1) < change_probability:
                    image = cv2.flip(image, random.randint(-1, 1))
                image = image.reshape(input_size[0], input_size[1], 1)
                data_x.append(image)
                data_y.append (labels[i])
        
        hf = h5py.File(h5file, 'w')
        hf.create_dataset('data_x', data=data_x, compression="gzip")
        hf.create_dataset('data_y', data=data_y, compression="gzip")
        hf.create_dataset('augment_list', data=augment_list, compression="gzip")
        hf.close()

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    return data_x, data_y, augment_list

def get_model(differential_privacy=False, noise_multiplier=1.4, federated=False):
    
    model = Sequential()

    model.add(Conv2D(64, filters, input_shape=(input_size[0], input_size[1], 1), padding='same',
                     kernel_regularizer=l1_l2(regularizers, regularizers), activation='relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(64, filters, kernel_regularizer=l2(regularizers), padding='same',
               activation='relu'))
    # model.add(MaxPooling2D())
    model.add(MaxPooling2D((4, 4)))
    model.add(Dropout(0.20))

    model.add(
        Conv2D(256, filters, kernel_regularizer=l2(regularizers), padding='same',
               activation='relu'))
    model.add(
        Conv2D(256, filters, kernel_regularizer=l2(regularizers), padding='same',
               activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(Flatten())
    # model.add(Dense(512, kernel_regularizer=l2(regularizers), activation='relu'))
    model.add(Dense(256, kernel_regularizer=l2(regularizers), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    # model.load_weights("weights.best.hdf5")

    # opt = Adam(lr=0.001)

    optimizer = Adam()
    if differential_privacy:
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)

    if not federated:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model



def main(winL=90, winR=90, do_preprocess=True, 
    maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, reduced_DS = False, leads_flag = [1,0],
    differential_privacy=False, noise_multiplier=1.4):
    
    db_path = settings.db_path
    
    # Load train data 
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
        maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag)

    # # Load test data
    # [eval_features, eval_labels, eval_patient_num_beats] = load_mit_db('DS2', winL, winR, do_preprocess, 
    #     maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag)

    # scaler = StandardScaler()
    # scaler.fit(tr_features)
    # tr_features_scaled = scaler.transform(tr_features)
    # eval_features_scaled = scaler.transform(eval_features)

    # [train_x, train_y] = get_images_data(tr_features, tr_labels, 1000, "train")
    # [train_x, train_y] = get_images_data(tr_features, tr_labels, tr_features.shape[0], "train")
    # [test_x, test_y] = get_images_data(eval_features, eval_labels, 1000, "test")
    # [test_x, test_y] = get_images_data(eval_features, eval_labels, eval_features.shape[0], "test")

    [data_x, data_y, _] = get_images_data(tr_features, tr_labels, 20000, "train")

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_split, random_state=40)

    train_y = to_categorical(train_y, num_classes=n_classes)
    test_y = to_categorical(test_y, num_classes=n_classes)

    model_path = db_path + 'keras_cnn_models/'

    model_path = create_model_name(model_path, winL, winR, do_preprocess,
        maxRR, use_RR, norm_RR, compute_morph,
        leads_flag, reduced_DS, '_')

    model_path = model_path + '.h5'

    print("Training model on MIT-BIH DS1: " + model_path + "...")

    if 1==2:#os.path.isfile(model_svm_path):
        # Load the trained model!
        model = load_model(model_path)

    else:

        model = get_model(differential_privacy, noise_multiplier)
        model.summary()

        checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # Let's Train!
        start = time.time()
        model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_split=validation_split, callbacks=callbacks_list)
        end = time.time()

        print(("Trained completed!\n\t" + model_path + "\n \
            \tTime required: " + str(format(end - start, '.2f')) + " sec" ))

        # Save trained MLP model
        model.save(model_path)

    # Test the model
    print(("Testing model on MIT-BIH DS2: " + model_path + "..."))

    # Evaluate the model with new data
    predictions = model.predict(test_x)
    predictions = predictions > 0.5
    print(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(predictions, axis=1)))
    print(classification_report(test_y, predictions))
    print("Accuracy: {0}".format(accuracy_score(test_y, predictions)))

    if differential_privacy:
        eps = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=train_x.shape[0], batch_size=batch_size, noise_multiplier=noise_multiplier, epochs=epochs, delta=4e-6)
        with open("dp.txt", "a+") as f:
            f.write("noise={0} eps={1} training_time={2:.0f} s \n".format(noise_multiplier, eps, end - start))
            f.write(np.array2string(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(predictions, axis=1))))
            f.write(classification_report(test_y, predictions))
            f.write("Accuracy: {0} \n".format(accuracy_score(test_y, predictions)))
            f.write("-------------------------\n")


def update_num_beats(patient_num_beats, augment_list):
    cumulative_list = [np.sum(patient_num_beats[:i+1]) for i in range(NUM_CLIENTS)]
    lst = np.array(augment_list)
    for i in range(NUM_CLIENTS):
        result = np.where((lst > (i > 0)*cumulative_list[i-1]) & (lst <= cumulative_list[i]))
        patient_num_beats[i] = patient_num_beats[i] + 9 * result[0].size
    return patient_num_beats



def main_federated(winL=90, winR=90, do_preprocess=True, 
    maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, reduced_DS = False, leads_flag = [1,0]):
    print("Runing train_Keras.py for federated learning!")

    db_path = settings.db_path
    
    # Load train data 
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
        maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag)

    [data_x, data_y, augment_list] = get_images_data(tr_features, tr_labels, np.sum(tr_patient_num_beats[:NUM_CLIENTS]), "train_fed")

    data_y = to_categorical(data_y, num_classes=n_classes)

    tr_patient_num_beats = update_num_beats(tr_patient_num_beats, augment_list)

    # clients_x = [np.array(data_x[i:i+tr_patient_num_beats[i]], dtype=np.single) for i in range(NUM_CLIENTS)]
    clients_x = [np.array(data_x[(i > 0)*np.sum(tr_patient_num_beats[:i]):np.sum(tr_patient_num_beats[:i+1])], dtype=np.single) 
                    for i in range(NUM_CLIENTS)]
    # clients_y = [np.array(data_y[i:i+tr_patient_num_beats[i]], dtype=np.single) for i in range(NUM_CLIENTS)]
    clients_y = [np.array(data_y[(i > 0)*np.sum(tr_patient_num_beats[:i]):np.sum(tr_patient_num_beats[:i+1])], dtype=np.single) 
                    for i in range(NUM_CLIENTS)]

    clients_split = [ train_test_split(clients_x[i], clients_y[i], test_size=test_split, random_state=40) for i in range(NUM_CLIENTS)]

    train_clients = np.array([np.array(clients_split[i][0]) for i in range(NUM_CLIENTS)])
    test_clients = np.array([np.array(clients_split[i][1]) for i in range(NUM_CLIENTS)])
    train_labels_clients = np.array([np.array(clients_split[i][2]) for i in range(NUM_CLIENTS)])
    test_labels_clients = np.array([np.array(clients_split[i][3]) for i in range(NUM_CLIENTS)])

    def preprocess(dataset):
        return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER)

    def make_federated_data(client_data, client_labels_data, client_ids):
        return [
            preprocess(tf.data.Dataset.from_tensor_slices((client_data[x], client_labels_data[x])))
            for x in client_ids
        ]

    sample_clients = range(NUM_CLIENTS)
    federated_train_data = make_federated_data(train_clients, train_labels_clients, sample_clients)

    def model_fn():
        keras_model = get_model(federated=True)
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=federated_train_data[0].element_spec,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    state = iterative_process.initialize()

    for round_num in range(2, NUM_ROUNDS+1):
        state, metrics = iterative_process.next(state, federated_train_data)
        print('round {:2d}, metrics={}'.format(round_num, metrics))
        with open("rounds.txt", "a+") as f:
            f.write('round {:2d}, metrics={}\n'.format(round_num, metrics))
            
    evaluation = tff.learning.build_federated_evaluation(model_fn)

    train_metrics = evaluation(state.model, federated_train_data)
    print("Train metrics: ")
    print(str(train_metrics))

    federated_test_data = make_federated_data(test_clients, test_labels_clients, sample_clients)

    test_metrics = evaluation(state.model, federated_test_data)
    print("Test metrics: ")
    print(str(test_metrics))

    model = get_model()
    tff.learning.assign_weights_to_keras_model(model, state.model)
    model.save("fed_model.h5")

    test_x = test_clients[0]
    for i in range(1, NUM_CLIENTS):
        test_x = np.concatenate((test_x, test_clients[i]))

    test_y = test_labels_clients[0]
    for i in range(1, NUM_CLIENTS):
        test_y = np.concatenate((test_y, test_labels_clients[i]))
    
    predictions = model.predict(test_x)
    predictions = predictions > 0.5
    print(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(predictions, axis=1)))
    print(classification_report(test_y, predictions))
    print("Accuracy: {0}".format(accuracy_score(test_y, predictions)))