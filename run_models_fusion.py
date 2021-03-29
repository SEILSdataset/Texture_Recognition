#!/usr/bin/env python
# -- coding: utf-8 --
import os
import copy
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import shutil
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from operator import itemgetter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, BatchNormalization, Activation, LSTM, Bidirectional, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras_self_attention import SeqSelfAttention
import warnings
warnings.filterwarnings('ignore')


def LSTM_model(conf_lstm, net):
    for ind, layers in enumerate(conf_lstm):
        if ind == len(conf_lstm) - 1:
            return_sequences = False  # Final LSTM layer outputs only one vector ...
        else:
            return_sequences = True  # ... and previous layers output a sequence
        net = Bidirectional(LSTM(150, dropout=0.2, return_sequences=return_sequences, activation='tanh'))(net)
        if return_sequences:
            net = SeqSelfAttention(attention_activation='sigmoid')(net)
    return net


def CNN_model(conf_cnn, net):
    for layers in conf_cnn:
        net = Conv1D(150, (3,), strides=2, activation='linear', padding='same')(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = MaxPool1D(2, strides=2, padding='same')(net)
    net = GlobalMaxPool1D()(net)
    return net


def fill_dics_fold(X_dic, Y_dic, elem, features, target):
    X_dic[elem] = features[elem]
    Y_dic[elem] = target
    return X_dic, Y_dic


def partitioning_LLDs(num_class, features, A, B, C, index):
    features_3T = {}
    features_2T = {}

    for elem in features:
        if num_class == 3 and 'MIX' not in elem:
            features_3T[elem] = features[elem]
        elif num_class == 2 and 'MIX' not in elem and 'ANT' not in elem:
            features_2T[elem] = features[elem]

    if num_class == 3:
        features = features_3T
    elif num_class == 2:
        features = features_2T

    X_A = {}
    X_B = {}
    X_C = {}
    Y_A = {}
    Y_B = {}
    Y_C = {}

    for elem in features:
        a, composer, target = elem.split('_')
        if composer in A:
            X_A, Y_A = fill_dics_fold(X_A, Y_A, elem, features, target)
        elif composer in B:
            X_B, Y_B = fill_dics_fold(X_B, Y_B, elem, features, target)
        elif composer in C:
            X_C, Y_C = fill_dics_fold(X_C, Y_C, elem, features, target)

    if index <= 1:
        X_A, Y_A = make_upsampling(X_A, Y_A, num_class)
    elif index <= 3:
        X_B, Y_B = make_upsampling(X_B, Y_B, num_class)
    else:
        X_C, Y_C = make_upsampling(X_C, Y_C, num_class)

    ALL_dics = {'X_A': X_A, 'X_B': X_B, 'X_C': X_C, 'Y_A': Y_A, 'Y_B': Y_B, 'Y_C': Y_C}

    return ALL_dics


def NN_model(conf_cnn_lstm, conf_mlp, classifier, num_class, part_name, num_feat):
    # add input layer
    if part_name == 'All':
        input_llds = Input(shape=(None, 5, num_feat))
        net = Reshape((-1, num_feat*5))(input_llds)
    else:
        input_llds = Input(shape=(None, num_feat))
        net = input_llds

    # add CNN/LSTM
    if classifier == 'CNN':
        net = CNN_model(conf_cnn_lstm, net)
    else:
        net = LSTM_model(conf_cnn_lstm, net)

    # concatenate with functionals
    if num_feat == 9:
        N_functionals = 163
    elif num_feat == 11:
        N_functionals = 195
    input_func    = Input(shape=(N_functionals,))
    net           = Concatenate()([net, input_func])

    # add the hidden layers with relu activation function
    counter = 0
    for neurons in conf_mlp:
        counter = counter + 1
        net = Dense(neurons, activation="sigmoid")(net)
        if counter < len(conf_mlp):
            net = Dropout(0.2)(net)

    # add output layer with N neurons = N classes and softmax activation function
    net   = Dense(num_class, activation="softmax")(net)
    model = Model(inputs=[input_llds, input_func], outputs=net)

    return model


def perform_CNN_LSTM(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, num_class, N_samples_fold, classifier, part_name, dic_train_MLP, dic_dev_MLP, dic_test_MLP, num_feat):
    results                 = []
    configurations_mlp      = [(25,25), (75,75), (175, 175), ]
    learning_rates          = [0.001, ]
    configurations_cnn_lstm = [[1,], [1,2]]
    batch_sizes             = [50, 25, 10]
    best_UAR                = 0.
    le                      = LabelEncoder()  # convert labels into numeric encoding

    # for TRAIN #
    numeric_target      = le.fit_transform(Y_train)
    onehot_target_train = to_categorical(numeric_target)  # convert to one-hot target encoding

    # for DEV #
    numeric_target    = le.transform(Y_dev)
    onehot_target_dev = to_categorical(numeric_target)  # convert to one-hot target encoding

    for conf_cnn_lstm in configurations_cnn_lstm:
        for conf_mlp in configurations_mlp:
            for learn in learning_rates:
                for batch_size in batch_sizes:
                    tf.keras.backend.clear_session()
                    gc.collect()  # clear RAM
                    np.random.seed(1)
                    tf.random.set_seed(1)
                    # build the model
                    model     = NN_model(conf_cnn_lstm, conf_mlp, classifier, num_class, part_name, num_feat)
                    optimizer = Adam(lr=learn)
                    # compile the model
                    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
                    callback = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
                    print("Dev model")
                    print(model.summary())
                    # fit the model
                    model.fit([X_train, dic_train_MLP['X']], onehot_target_train, validation_data=([X_dev, dic_dev_MLP['X']], onehot_target_dev), batch_size=batch_size, shuffle=True, epochs=400, callbacks=[callback])  #HERE#
                    predictions = model.predict([X_dev, dic_dev_MLP['X']], batch_size=batch_size)
                    predictions = np.argmax(predictions, axis=1)  # convert the predictions' labels from one-hot to categorical
                    predictions = le.inverse_transform(predictions)
                    UAR         = recall_score(Y_dev, predictions, average='macro')

                    if UAR > best_UAR:
                        best_UAR = UAR
                        best_model = tf.keras.models.clone_model(model)
                        best_model.set_weights(model.get_weights())

                    results.append((conf_cnn_lstm, conf_mlp, learn, batch_size, best_UAR))

    # Get best configuration on the development
    best_Conf   = max(results, key=itemgetter(4))[0:4]
    predictions = best_model.predict([X_test, dic_test_MLP['X']], batch_size=batch_size)
    predictions = np.argmax(predictions, axis=1)
    predictions = le.inverse_transform(predictions)

    return predictions, best_Conf, Y_test


def concat_arrays(my_arrays, part_name):
    if part_name == 'All':
        balance_part_arrays = [np.concatenate((elem, np.zeros((elem.shape[0], 5 - elem.shape[1], elem.shape[2]))), axis=1) for elem in my_arrays]
        max_entries = max([len(x) for x in balance_part_arrays])
        my_arrays3 = [np.concatenate((elem, np.zeros((max_entries - len(elem), elem.shape[1], elem.shape[2])))) for elem in balance_part_arrays]
    else:
        max_entries = max([len(x) for x in my_arrays])
        my_arrays3 = [np.concatenate((elem, np.zeros((max_entries - len(elem), elem.shape[1])))) for elem in my_arrays]
    feature_4D = np.stack(my_arrays3, axis=0)
    return feature_4D


def feature_normalisation(dic_train_x, dic_dev_x, dic_test_x, dic_train_y, dic_dev_y, dic_test_y, part_name, num_feat):
    X_train = [dic_train_x[elem] for elem in dic_train_x]
    Y_train = [dic_train_y[elem] for elem in dic_train_x]
    X_dev   = [dic_dev_x[elem] for elem in dic_dev_x]
    Y_dev   = [dic_dev_y[elem] for elem in dic_dev_x]
    X_test  = [dic_test_x[elem] for elem in dic_test_x]
    Y_test  = [dic_test_y[elem] for elem in dic_test_x]
    # add 0 to match arrays' shape in parts and global time dimension (X = 4D array; Y = 1D array)
    X_train = concat_arrays(X_train, part_name)
    Y_train = np.array(Y_train)
    X_dev   = concat_arrays(X_dev, part_name)
    Y_dev   = np.array(Y_dev)
    X_test  = concat_arrays(X_test, part_name)
    Y_test  = np.array(Y_test)

    # NORMALISE
    scaler         = StandardScaler()
    reshaped_train = scaler.fit_transform(X_train.reshape(-1, num_feat))  # reshape to 2D and apply feature normalisation (-1 means that the first three dim are stacked together):
    X_train        = reshaped_train.reshape(X_train.shape)  # reshape back the normalised features into 4D

    # transform dev
    reshaped_dev = scaler.transform(X_dev.reshape(-1, num_feat))
    X_dev        = reshaped_dev.reshape(X_dev.shape)

    # transform test
    reshaped_test = scaler.transform(X_test.reshape(-1, num_feat))
    X_test        = reshaped_test.reshape(X_test.shape)

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test


def run_experiments(ALL_dics, folds, classifier, num_class, part_name, dic_train_MLP, dic_dev_MLP, dic_test_MLP, num_feat):
    # FEATURE NORMALISATION
    new_dict = copy.deepcopy(ALL_dics)
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = feature_normalisation(new_dict['X' + folds[0]], new_dict['X' + folds[1]], new_dict['X' + folds[2]], new_dict['Y' + folds[0]], new_dict['Y' + folds[1]], new_dict['Y' + folds[2]], part_name, num_feat)
    shape_A = Y_train.shape
    shape_B = Y_dev.shape
    shape_C = Y_test.shape
    N_samples_fold = (shape_A[0] + shape_B[0] + shape_C[0])//3

    # RUN CLASSIFIER
    predictions, best_Conf, Y_test = perform_CNN_LSTM(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, num_class, N_samples_fold, classifier, part_name, dic_train_MLP, dic_dev_MLP, dic_test_MLP, num_feat)
    return predictions, best_Conf, Y_test


def evaluation(predictions, dic_test, UAR, WAR, rec_ANT, rec_CON, rec_HOM, rec_MIX, pre_ANT, pre_CON, pre_HOM, pre_MIX, conf_matrix, num_class):
    UAR = UAR + recall_score(dic_test, predictions, average='macro')
    WAR = WAR + recall_score(dic_test, predictions, average='weighted')

    if num_class == 4:
        cm = confusion_matrix(dic_test, predictions, labels=['CON', 'HOM', 'ANT', 'MIX'])
    elif num_class == 3:
        cm = confusion_matrix(dic_test, predictions, labels=['CON', 'HOM', 'ANT'])
    else:
        cm = confusion_matrix(dic_test, predictions, labels=['CON', 'HOM'])

    percent_cm  = get_CM_percent(cm)
    conf_matrix = conf_matrix + percent_cm
    rec_CON     = rec_CON + percent_cm[0, 0]
    rec_HOM     = rec_HOM + percent_cm[1, 1]
    pre_CON     = pre_CON + (np.divide(int(cm[0, 0]), int(np.sum(cm[:, 0])), where=int(np.sum(cm[:, 0])) > 0))*100
    pre_HOM     = pre_HOM + (np.divide(int(cm[1, 1]), int(np.sum(cm[:, 1])), where=int(np.sum(cm[:, 1])) > 0))*100

    if num_class > 2:
        rec_ANT = rec_ANT + percent_cm[2, 2]
        pre_ANT = pre_ANT + (np.divide(int(cm[2, 2]), int(np.sum(cm[:, 2])), where=int(np.sum(cm[:, 2])) > 0))*100
        if num_class == 4:
            rec_MIX = rec_MIX + percent_cm[3, 3]
            pre_MIX = pre_MIX + (np.divide(int(cm[3, 3]), int(np.sum(cm[:, 3])), where=int(np.sum(cm[:, 3])) > 0))*100

    return UAR, WAR, rec_ANT, rec_CON, rec_HOM, rec_MIX, pre_ANT, pre_CON, pre_HOM, pre_MIX, conf_matrix


def get_csv(predictions, all_results, dic_test, num_class, classifier):
    UAR_f = recall_score(dic_test, predictions, average='macro')

    if num_class == 4:
        classes = ['CON', 'HOM', 'ANT', 'MIX']
    elif num_class == 3:
        classes = ['CON', 'HOM', 'ANT']
    else:
        classes = ['CON', 'HOM']

    rec_result  = recall_score(dic_test, predictions, average=None, labels=classes)
    prec_result = precision_score(dic_test, predictions, average=None, labels=classes)
    rec_CON_f   = rec_result[0]
    rec_HOM_f   = rec_result[1]
    pre_CON_f   = prec_result[0]
    pre_HOM_f   = prec_result[1]

    if num_class > 2:
        rec_ANT_f = rec_result[2]
        pre_ANT_f = prec_result[2]
        if num_class == 4:
            rec_MIX_f = rec_result[3]
            pre_MIX_f = prec_result[3]

    if not classifier + '_UAR' in all_results:
        all_results[classifier + '_UAR'] = [UAR_f]
    else:
        all_results[classifier + '_UAR'].append(UAR_f)

    metric = ['rec', 'pre']
    for mad_class in classes:
        for unit in metric:
            val = eval(unit + '_' + mad_class + '_f')
            if not classifier + '_' + unit + '_' + mad_class in all_results:
                all_results[classifier + '_' + unit + '_' + mad_class] = [val]
            else:
                all_results[classifier + '_' + unit + '_' + mad_class].append(val)

    return all_results


def make_folds():
    all_folds = [('_A', '_B', '_C'),
                 ('_A', '_C', '_B'),
                 ('_B', '_A', '_C'),
                 ('_B', '_C', '_A'),
                 ('_C', '_B', '_A'),
                 ('_C', '_A', '_B')]
    return all_folds


def get_CM_percent(conf_matrix):
    print(conf_matrix)
    conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1).reshape(-1, 1)
    return conf_matrix


def make_withinfold_upsampling_MLP(dic, num_class):
    # upsampling minority classes
    df1 = pd.DataFrame(dic['X'])
    df2 = pd.DataFrame(dic['Y'], columns=['annotation'])
    df3 = pd.concat([df1, df2], axis=1)
    top = max(df3['annotation'].value_counts())

    missing_HOM = top - df3['annotation'].value_counts()['HOM']
    HOM_list = []

    if num_class == 3:
        ANT_list = []
        missing_ANT = top - df3['annotation'].value_counts()['ANT']
        features = df3
        while len(ANT_list) < missing_ANT:
            for row in features.itertuples(index=False):
                if row.annotation == 'HOM' and len(HOM_list) < missing_HOM:
                    HOM_list.append(list(row))
                elif row.annotation == 'ANT' and len(ANT_list) < missing_ANT:
                    ANT_list.append(list(row))
        df  = pd.DataFrame(HOM_list, columns=features.columns)
        df1 = pd.DataFrame(ANT_list, columns=features.columns)
        df3 = pd.concat([df, df3])
        df3 = pd.concat([df1, df3])
        dic['Y'] = df3.loc[:, ['annotation']].to_numpy()
        dic['X'] = df3.loc[:, df.columns != 'annotation'].to_numpy()
    else:
        features = df3
        while len(HOM_list) < missing_HOM:
            for row in features.itertuples(index=False):
                if (row.annotation == 'HOM') and (len(HOM_list) < missing_HOM):
                    HOM_list.append(list(row))
        df = pd.DataFrame(HOM_list, columns=features.columns)
        df4 = pd.concat([df, df3])
        dic['Y'] = df4.loc[:, ['annotation']].to_numpy()
        dic['X'] = df4.loc[:, df.columns != 'annotation'].to_numpy()

    return dic


def run_main_function(results, dic_features, num_class, classifier, all_results, part_name, A, B, C, ALL_dics_MLP, N_functionals, num_feat):
    all_folds     = make_folds()
    all_folds_MLP = make_folds_MLP('all_flat')
    UAR     = 0
    WAR     = 0
    rec_ANT = 0
    rec_HOM = 0
    rec_CON = 0
    rec_MIX = 0
    pre_ANT = 0
    pre_HOM = 0
    pre_CON = 0
    pre_MIX = 0
    conf_matrix = np.zeros(shape=(num_class, num_class))

    for index, folds in enumerate(all_folds):
        ALL_dics = partitioning_LLDs(num_class, dic_features, A, B, C, index)
        dic_train_MLP, dic_dev_MLP, dic_test_MLP, N_samples_fold = get_dics_MLP(ALL_dics_MLP, all_folds_MLP[index], N_functionals, num_class)
        dic_train_MLP = make_withinfold_upsampling_MLP(dic_train_MLP, num_class)
        predictions, best_Conf, Y_test = run_experiments(ALL_dics, folds, classifier, num_class, part_name, dic_train_MLP, dic_dev_MLP, dic_test_MLP, num_feat)
        results.append(best_Conf)
        all_results = get_csv(predictions, all_results, Y_test, num_class, classifier)
        UAR, WAR, rec_ANT, rec_CON, rec_HOM, rec_MIX, pre_ANT, pre_CON, pre_HOM, pre_MIX, conf_matrix = evaluation(predictions, Y_test, UAR, WAR, rec_ANT, rec_CON, rec_HOM, rec_MIX, pre_ANT, pre_CON, pre_HOM, pre_MIX, conf_matrix, num_class)

    results.append('UAR = ' + str(UAR/6))
    results.append('WAR = ' + str(WAR/6))
    results.append('Recall for CON = ' + str(rec_CON/6))
    results.append('Recall for HOM = ' + str(rec_HOM/6))
    results.append('Recall for ANT = ' + str(rec_ANT/6))
    results.append('Recall for MIX = ' + str(rec_MIX/6))
    results.append('Precision for CON = ' + str(pre_CON/6))
    results.append('Precision for HOM = ' + str(pre_HOM/6))
    results.append('Precision for ANT = ' + str(pre_ANT/6))
    results.append('Precision for MIX = ' + str(pre_MIX/6))
    results.append(conf_matrix/6)

    for line in results:
        print(line)

    return results, all_results


def printintg_file(results, my_dir, classifier, num_class, i, part_name, num_feat):
    f = open(my_dir + '/all_RESULTS_FUSION_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/' + classifier + "_RESULTS_" + part_name + str(i) + ".txt", "w")
    print('Speceific results for each random splitting')

    for my_elem in results:
        if isinstance(my_elem, np.ndarray):
            for column in my_elem:
                if num_class == 4:
                    print("{} {} {} {}".format(column[0], column[1], column[2], column[3]), file=f)
                elif num_class == 3:
                    print("{} {} {}".format(column[0], column[1], column[2]), file=f)
                elif num_class == 2:
                    print("{} {}".format(column[0], column[1]), file=f)
        else:
            print(my_elem, file=f)
            print(my_elem)
    f.close()


def get_random_split(split, my_dir):
    random.seed(split)
    print('SPLITTING IN 3-FOLDS')
    path      = my_dir + '/corpus'
    composers = []

    for krn_file in glob.glob(os.path.join(path, '*.krn')):
        file_name = os.path.basename(krn_file[0:-4])
        composer, other = file_name.split('_', 1)
        composers.append(composer)
    print(composers)
    random.shuffle(composers)
    print(composers)
    A = composers[0:10]
    B = composers[10:20]
    C = composers[20:30]

    return A, B, C


def make_upsampling(x, y, num_class):
    # upsampling minority classes
    mad_list      = []
    composer_list = []

    for elem in list(y.keys()):
        num, composer_name, mad_class = elem.split('_')
        mad_list.append(mad_class)
        composer_list.append(composer_name)

    counter = [('CON', mad_list.count('CON')), ('HOM', mad_list.count('HOM')), ('ANT', mad_list.count('ANT')), ('MIX', mad_list.count('MIX'))]
    top     = max(counter, key=itemgetter(1))[1]

    missing_HOM = top - counter[list(map(itemgetter(0), counter)).index('HOM')][1]
    missing_CON = top - counter[list(map(itemgetter(0), counter)).index('CON')][1]
    missing_ANT = top - counter[list(map(itemgetter(0), counter)).index('ANT')][1]

    new_HOM_x = {}
    new_CON_x = {}
    new_ANT_x = {}
    new_HOM_y = {}
    new_CON_y = {}
    new_ANT_y = {}
    add_num   = 99

    if num_class == 3:
        while len(new_ANT_x) < missing_ANT:
            add_num = add_num + 1
            for index, mad in enumerate(mad_list):
                if mad == 'HOM' and len(new_HOM_x) < missing_HOM:
                    new_HOM_x[str(add_num) + list(x.keys())[index]] = x[list(x.keys())[index]]
                    new_HOM_y[str(add_num) + list(y.keys())[index]] = y[list(y.keys())[index]]
                elif mad == 'ANT' and len(new_ANT_x) < missing_ANT:
                    new_ANT_x[str(add_num) + list(x.keys())[index]] = x[list(x.keys())[index]]
                    new_ANT_y[str(add_num) + list(y.keys())[index]] = y[list(y.keys())[index]]
                elif mad == 'MIX' and len(new_CON_x) < missing_CON:
                    new_CON_x[str(add_num) + list(x.keys())[index]] = x[list(x.keys())[index]]
                    new_CON_y[str(add_num) + list(y.keys())[index]] = y[list(y.keys())[index]]
        merged_x = {**new_ANT_x, **new_HOM_x, **new_CON_x}
        x.update(merged_x)
        merged_y = {**new_ANT_y, **new_HOM_y, **new_CON_y}
        y.update(merged_y)
    else:
        while len(new_HOM_x) < missing_HOM:
            add_num = add_num + 1
            for index, mad in enumerate(mad_list):
                if mad == 'HOM' and len(new_HOM_x) < missing_HOM:
                    new_HOM_x[str(add_num) + list(x.keys())[index]] = x[list(x.keys())[index]]
                    new_HOM_y[str(add_num) + list(y.keys())[index]] = y[list(y.keys())[index]]
                elif mad == 'MIX' and len(new_CON_x) < missing_CON:
                    new_CON_x[str(add_num) + list(x.keys())[index]] = x[list(x.keys())[index]]
                    new_CON_y[str(add_num) + list(y.keys())[index]] = y[list(y.keys())[index]]
        merged_x = {**new_HOM_x, **new_CON_x}
        x.update(merged_x)
        merged_y = {**new_HOM_y, **new_CON_y}
        y.update(merged_y)

    return x, y


def get_features(my_dir, num_feat):
    features   = {}
    features_C = {}
    features_A = {}
    features_Q = {}
    features_T = {}
    features_B = {}
    f = open(my_dir + "/folders_order.txt", "r")
    if num_feat == 11:
        LLDs_deltas = ['note_pitchPS', 'text_mus', 'interval_num', 'rhythm', 'offset', 'beat_num', 'beat_num_delta', 'note_pitchPS_delta', 'interval_num_delta', 'rhythm_delta', 'offset_delta']
    elif num_feat == 9:
        LLDs_deltas = ['text_mus', 'interval_num', 'rhythm', 'offset', 'beat_num', 'beat_num_delta', 'interval_num_delta', 'rhythm_delta', 'offset_delta']
    
    for line in f:
        elem = line.rstrip()
        my_arrays = []
        for part in os.listdir(my_dir + '/LLD_Deltas_all2/' + elem):
            if part != 'all_flat.csv':
                features_part = pd.DataFrame.to_numpy(pd.read_csv(my_dir + '/LLD_Deltas_all2/' + elem + '/' + part, sep='\t', engine='python')[LLDs_deltas])
                my_arrays.append(features_part)
                if part == 'Canto.csv':
                    features_C[elem] = features_part
                elif part == 'Alto.csv':
                    features_A[elem] = features_part
                elif part == 'Quinto.csv':
                    features_Q[elem] = features_part
                elif part == 'Tenor.csv':
                    features_T[elem] = features_part
                elif part == 'Bass.csv':
                    features_B[elem] = features_part
        # add 0 to match arrays' shape
        max_entries    = max([len(x) for x in my_arrays])
        my_arrays2     = [np.concatenate((elem, np.zeros((max_entries - len(elem), elem.shape[1])))) for elem in my_arrays]
        feature_3D     = np.stack(my_arrays2, axis=1)
        features[elem] = feature_3D
    
    return features, features_C, features_A, features_Q, features_T, features_B


def fill_array(dic, index, X, Y, Z):
    dic['X'] = np.append(dic['X'], [X[index]], axis=0)
    dic['Y'] = np.append(dic['Y'], [Y[index]], axis=0)
    dic['Z'] = np.append(dic['Z'], [Z[index]], axis=0)
    return dic


def fill_part_array(all_dic, all_flat_dic, canto_dic, alto_dic, quinto_dic, tenor_dic, bass_dic):
    for index, elem in enumerate(all_dic['Z'][:, -1:]):
        if elem == 'all_flat':
            all_flat_dic = fill_array(all_flat_dic, index, all_dic['X'], all_dic['Y'], all_dic['Z'])
        elif elem == 'Canto':
            canto_dic = fill_array(canto_dic, index, all_dic['X'], all_dic['Y'], all_dic['Z'])
        elif elem == 'Alto':
            alto_dic = fill_array(alto_dic, index, all_dic['X'], all_dic['Y'], all_dic['Z'])
        elif elem == 'Quinto':
            quinto_dic = fill_array(quinto_dic, index, all_dic['X'], all_dic['Y'], all_dic['Z'])
        elif elem == 'Tenor':
            tenor_dic = fill_array(tenor_dic, index, all_dic['X'], all_dic['Y'], all_dic['Z'])
        elif elem == 'Bass':
            bass_dic = fill_array(bass_dic, index, all_dic['X'], all_dic['Y'], all_dic['Z'])
    return all_flat_dic, canto_dic, alto_dic, quinto_dic, tenor_dic, bass_dic


def partitioning_functionals(num_class, features, A, B, C, list_excluded2):
    if num_class == 3:
        features = features.loc[features['annotation'].isin(['CON', 'HOM', 'ANT'])]
    elif num_class == 2:
        features = features.loc[features['annotation'].isin(['CON', 'HOM'])]

    # features selection
    list_excluded1 = ['Unnamed: 0', 'annotation', 'annotation_ID', 'composer', 'part_name', 'part_clef']
    features_selected = pd.DataFrame()
    for feature in features.columns:
        if (feature not in list_excluded1) and (feature not in list_excluded2):
            features_selected.insert(len(features_selected.columns), feature, features[feature])
    X = features_selected.values

    print('N of functionals: ', X.shape[1])
    N_functionals = X.shape[1]
    print('Total N of samples: ', X.shape[0])

    Y = features.loc[:, ['annotation']].to_numpy()
    Z = features.loc[:, ['composer', 'part_name']].to_numpy()

    print('List of madrigalisms: ', np.unique(Y))
    print(features['annotation'].value_counts())
    print('List of parts: ', np.unique(Z[:, -1:]))

    # PARTITIONING
    all_A = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    all_B = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    all_C = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}

    # Generate the ALL dic: make a loop to fill out the numpy arrays in base on the criterion composer (arrays are filled out over the 0 axis, i.e. the rows)
    for index, elem in enumerate(Z[:, :1]):
        if elem in A:
            all_A = fill_array(all_A, index, X, Y, Z)
        elif elem in B:
            all_B = fill_array(all_B, index, X, Y, Z)
        else:
            all_C = fill_array(all_C, index, X, Y, Z)

    # generate the PARTS dic:
    all_flat_A = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    all_flat_B = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    all_flat_C = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}

    canto_A = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    canto_B = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    canto_C = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}

    alto_A = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    alto_B = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    alto_C = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}

    quinto_A = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    quinto_B = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    quinto_C = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}

    tenor_A = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    tenor_B = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    tenor_C = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}

    bass_A = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    bass_B = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}
    bass_C = {'X': np.empty((0, X.shape[1])), 'Y': np.empty((0, 1)), 'Z': np.empty((0, 2))}

    all_flat_A, canto_A, alto_A, quinto_A, tenor_A, bass_A = fill_part_array(all_A, all_flat_A, canto_A, alto_A, quinto_A, tenor_A, bass_A)
    all_flat_B, canto_B, alto_B, quinto_B, tenor_B, bass_B = fill_part_array(all_B, all_flat_B, canto_B, alto_B, quinto_B, tenor_B, bass_B)
    all_flat_C, canto_C, alto_C, quinto_C, tenor_C, bass_C = fill_part_array(all_C, all_flat_C, canto_C, alto_C, quinto_C, tenor_C, bass_C)

    ALL_dics     = {}
    partitioning = ['A', 'B', 'C']
    parts        = ['all_flat', 'canto', 'alto', 'quinto', 'tenor', 'bass']

    for part in parts:
        for fold in partitioning:
            key_name = part + '_' + fold
            dic      = eval(key_name)
            ALL_dics[key_name] = dic

    return ALL_dics, N_functionals


def make_folds_MLP(data_group):
    all_folds = [(data_group + '_A', data_group + '_B', data_group + '_C'),
                 (data_group + '_A', data_group + '_C', data_group + '_B'),
                 (data_group + '_B', data_group + '_A', data_group + '_C'),
                 (data_group + '_B', data_group + '_C', data_group + '_A'),
                 (data_group + '_C', data_group + '_B', data_group + '_A'),
                 (data_group + '_C', data_group + '_A', data_group + '_B')]
    return all_folds


def feature_normalisation_MLP(dic_train, dic_dev, dic_test):
    scaler         = StandardScaler()
    dic_train['X'] = scaler.fit_transform(dic_train['X'])
    dic_test['X']  = scaler.transform(dic_test['X'])  # Applies the same scaling and shifting operations performed on the train data
    dic_dev['X']   = scaler.transform(dic_dev['X'])  # Applies the same scaling and shifting operations performed on the train data
    return dic_train, dic_dev, dic_test


def get_dics_MLP(ALL_dics_MLP, folds, N_functionals, num_class):
    # FEATURE NORMALISATION
    new_dict = copy.deepcopy(ALL_dics_MLP)
    new_dict[folds[0]], new_dict[folds[1]], new_dict[folds[2]] = feature_normalisation_MLP(new_dict[folds[0]], new_dict[folds[1]], new_dict[folds[2]])
    shape_A = new_dict[folds[0]]['Y'].shape
    shape_B = new_dict[folds[1]]['Y'].shape
    shape_C = new_dict[folds[2]]['Y'].shape
    N_samples_fold = (shape_A[0] + shape_B[0] + shape_C[0])//3
    return new_dict[folds[0]], new_dict[folds[1]], new_dict[folds[2]], N_samples_fold


def experiments_per_part(splits, num_class, dic_features, all_results, my_dir, classifier, part_name, num_feat, features_functionals):
    for split in splits:
        print('Running experiments with split: ', split)
        A, B, C = get_random_split(split, my_dir)
        results = []
        if num_feat == 9:
            selected_feature = ['PS_quartile1_delta', 'PS_quartile3_delta', 'PS_mode_delta', 'PS_median_delta', 'PS_harmonic_delta', 'PS_iqr_delta', 'PS_variance_delta', 'PS_gmean_delta', 'PS_variation_delta', 'PS_skewness_delta', 'PS_kurtosis_delta', 'note_pitchPS_mean_delta', 'note_pitchPS_std_delta', 'range_PS_delta', 'min_PS_delta', 'max_PS_delta', 'PS_quartile1', 'PS_quartile3', 'PS_mode', 'PS_median', 'PS_harmonic', 'PS_iqr', 'PS_variance', 'PS_gmean', 'PS_variation', 'PS_skewness', 'PS_kurtosis', 'note_pitchPS_mean', 'note_pitchPS_std', 'range_PS', 'min_PS', 'max_PS', 'ratio_class_note', 'num_pitchClass', 'note_RN_ratio', 'num_notes', 'num_rest', 'repeat_interval_ratio', 'num_interval', 'num_repeat_note', 'posit_negat_ratio', 'num_posit_interval', 'num_negat_interval']
        elif num_feat == 11:
            selected_feature = ['ratio_class_note', 'num_pitchClass', 'note_RN_ratio', 'num_notes', 'num_rest', 'repeat_interval_ratio', 'num_interval', 'num_repeat_note', 'posit_negat_ratio', 'num_posit_interval', 'num_negat_interval']
        
        ALL_dics_MLP, N_functionals = partitioning_functionals(num_class, features_functionals, A, B, C, selected_feature)
        results, all_results = run_main_function(results, dic_features, num_class, classifier, all_results, part_name, A, B, C, ALL_dics_MLP, N_functionals, num_feat)
        printintg_file(results, my_dir, classifier, num_class, split, part_name, num_feat)

    print('Saving results to file')
    df_results = pd.DataFrame.from_dict(all_results)
    print(df_results)
    print(df_results.mean(axis=0))
    df_results.to_csv(my_dir + '/all_RESULTS_FUSION_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/' + part_name + '_results.csv', sep='\t')


def set_up(classifier, num_class, num_feat):
    splits      = [41, 2, 25, 28]
    all_results = {}
    my_dir      = os.getcwd()

    if os.path.exists(my_dir + '/all_RESULTS_FUSION_' + classifier + '_' + str(num_class) + '_' + str(num_feat)):
        shutil.rmtree(my_dir + '/all_RESULTS_FUSION_' + classifier + '_' + str(num_class) + '_' + str(num_feat))
    os.mkdir(my_dir + '/all_RESULTS_FUSION_' + classifier + '_' + str(num_class) + '_' + str(num_feat))

    features_functionals = pd.read_csv(my_dir + '/functionals.csv', sep='\t')
    features, features_C, features_A, features_Q, features_T, features_B = get_features(my_dir, num_feat)
    experiments_per_part(splits, num_class, features, all_results, my_dir, classifier, 'All', num_feat, features_functionals)

    return my_dir


if __name__ == '__main__':
    classifier = 'CNN'
    
    for num_class in [2, 3]:
        for num_feat in [9, 11]:
            my_dir = set_up(classifier, num_class, num_feat)
            
            for elem in os.listdir(my_dir + '/all_RESULTS_FUSION_' + classifier + '_' + str(num_class) + '_' + str(num_feat)):
                print(elem)
                if '.txt' not in elem:
                    df_results = pd.read_csv(my_dir + '/all_RESULTS_FUSION_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/' + elem, sep='\t')
                    print(df_results.mean(axis=0))
