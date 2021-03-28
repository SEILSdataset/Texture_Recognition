#!/usr/bin/env python
# -- coding: utf-8 --
import os
import copy
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import shutil
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn import svm
from operator import itemgetter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')



def perform_MLP(dic_train, dic_dev, dic_test, N_functionals, num_class):
    dic_train   = make_withinfold_upsampling(dic_train, num_class)
    confs       = [(25,25), (75,75), (175, 175),]
    learn       = 0.001
    batch_sizes = [50, 25, 10]
    results     = []
    best_UAR    = 0.
    le          = LabelEncoder()  # convert labels into numeric encoding

    # for TRAIN #
    numeric_target      = le.fit_transform(dic_train['Y'])
    onehot_target_train = to_categorical(numeric_target)  # convert to one-hot target encoding

    # for DEV #
    numeric_target    = le.transform(dic_dev['Y'])
    onehot_target_dev = to_categorical(numeric_target)
    np.random.seed(1)
    tf.random.set_seed(1)
    
    for conf in confs:
        for batch in batch_sizes:
            # build the model
            model     = MLP_model(conf, N_functionals, num_class)
            optimizer = Adam(lr=learn)
        
            # compile the model
            model.compile(optimizer=optimizer, loss="categorical_crossentropy")
            print("Dev model")
            print(model.summary())
            callback = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        
            # fit the model
            model.fit(dic_train['X'], onehot_target_train, validation_data=(dic_dev['X'], onehot_target_dev), batch_size=batch, shuffle=True, epochs=200, callbacks=[callback])
            predictions = model.predict(dic_dev['X'], batch_size=batch)
            predictions = np.argmax(predictions, axis=1)  # convert the predictions' labels from one-hot to categorical
            predictions = le.inverse_transform(predictions)
            UAR         = recall_score(dic_dev['Y'], predictions, average='macro')
        
            if UAR > best_UAR:
                best_UAR   = UAR
                best_model = tf.keras.models.clone_model(model)
                best_model.set_weights(model.get_weights())
    
    results.append((conf, learn, batch, best_UAR))
    best_Conf   = max(results, key=itemgetter(3))[0:3]
    predictions = best_model.predict(dic_test['X'])
    predictions = np.argmax(predictions, axis=1)
    predictions = le.inverse_transform(predictions)

    return predictions, best_Conf


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
    list_excluded1    = ['Unnamed: 0', 'annotation', 'annotation_ID', 'composer', 'part_name', 'part_clef']
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


def MLP_model(conf, N_functionals, num_class):
    model = Sequential()
    model.add(Input(shape=(N_functionals,)))  # input layer with N neurons = N features
    counter = 0

    for neurons in conf:
        counter = counter + 1
        model.add(Dense(neurons, activation="sigmoid"))
        if counter < len(conf):
            model.add(Dropout(0.2))  # dropout except for the last layer

    model.add(Dense(num_class, activation="softmax"))  # output layer

    return model


def perform_SVM_3fold(dic_train, dic_dev, dic_test, num_class):
    dic_train = make_withinfold_upsampling(dic_train, num_class)
    C         = np.logspace(-5, -1, 5)  # Define levels of complexity: 5 values in logaritmic scale from 0.000001 to 0.1
    results   = []

    # run experiments for each complexity
    for elem in C:
        lin_clf = svm.LinearSVC(C=elem, random_state=42, dual=True)
        lin_clf.fit(dic_train['X'], dic_train['Y'].ravel())
        predictions = lin_clf.predict(dic_dev['X'])
        UAR         = recall_score(dic_dev['Y'], predictions, average='macro')
        results.append((UAR, elem))

    best_C = max(results, key=itemgetter(0))[1]  # Get best complexity from the cross-validation
    print('Performing test with complexity:')
    print(best_C)

    # Merge train and Dev
    merged_train_x = np.concatenate((dic_train['X'], dic_dev['X']))
    merged_train_y = np.concatenate((dic_train['Y'], dic_dev['Y']))

    # Make training again with the optimal hyper-parameters
    lin_clf = svm.LinearSVC(C=best_C, random_state=42)
    lin_clf.fit(merged_train_x, merged_train_y.ravel())

    return lin_clf.predict(dic_test['X']), best_C


def feature_normalisation(dic_train, dic_dev, dic_test):
    scaler         = StandardScaler()
    dic_train['X'] = scaler.fit_transform(dic_train['X'])
    dic_test['X']  = scaler.transform(dic_test['X'])  # Applies the same scaling and shifting operations performed on the train data
    dic_dev['X']   = scaler.transform(dic_dev['X'])  # Applies the same scaling and shifting operations performed on the train data
    return dic_train, dic_dev, dic_test


def run_experiments(ALL_dics, folds, classifier, N_functionals, num_class):
    # FEATURE NORMALISATION
    new_dict = copy.deepcopy(ALL_dics)
    new_dict[folds[0]], new_dict[folds[1]], new_dict[folds[2]] = feature_normalisation(new_dict[folds[0]], new_dict[folds[1]], new_dict[folds[2]])

    # RUN CLASSIFIER
    if classifier == 'SVM':
        predictions, best_Conf = perform_SVM_3fold(new_dict[folds[0]], new_dict[folds[1]], new_dict[folds[2]], num_class)
    elif classifier == 'MLP':
        predictions, best_Conf = perform_MLP(new_dict[folds[0]], new_dict[folds[1]], new_dict[folds[2]], N_functionals, num_class)

    return predictions, best_Conf, new_dict[folds[2]]['Y']


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


def make_folds(data_group):
    all_folds = [(data_group + '_A', data_group + '_B', data_group + '_C'),
                 (data_group + '_A', data_group + '_C', data_group + '_B'),
                 (data_group + '_B', data_group + '_A', data_group + '_C'),
                 (data_group + '_B', data_group + '_C', data_group + '_A'),
                 (data_group + '_C', data_group + '_B', data_group + '_A'),
                 (data_group + '_C', data_group + '_A', data_group + '_B')]
    return all_folds


def get_CM_percent(conf_matrix):
    print(conf_matrix)
    conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1).reshape(-1, 1)
    return conf_matrix


def run_main_function(data_group, results, ALL_dics, num_class, classifier, N_functionals, all_results):
    all_folds = make_folds(data_group)
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

    for folds in all_folds:
        predictions, best_Conf, dic_test = run_experiments(ALL_dics, folds, classifier, N_functionals, num_class)
        results.append(best_Conf)
        all_results = get_csv(predictions, all_results, dic_test, num_class, classifier)
        UAR, WAR, rec_ANT, rec_CON, rec_HOM, rec_MIX, pre_ANT, pre_CON, pre_HOM, pre_MIX, conf_matrix = evaluation(predictions, dic_test, UAR, WAR, rec_ANT, rec_CON, rec_HOM, rec_MIX, pre_ANT, pre_CON, pre_HOM, pre_MIX, conf_matrix, num_class)

    results.append(data_group)
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


def printintg_file(results, my_dir, classifier, num_class, i, elem, num_feat):
    f = open(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/' + classifier + elem + "_RESULTS_" + str(i) + ".txt", "w")
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


def get_random_split(features, split):
    random.seed(split)
    print('SPLITTING IN 3-FOLDS')
    composers = features['composer'].unique().tolist()
    print(composers)
    random.shuffle(composers)
    print(composers)
    A = composers[0:10]
    B = composers[10:20]
    C = composers[20:30]
    return A, B, C


def make_withinfold_upsampling(dic, num_class):
    # upsampling minority classes
    df1 = pd.DataFrame(dic['X'])
    df2 = pd.DataFrame(dic['Y'], columns=['annotation'])
    df3 = pd.concat([df1, df2], axis=1)
    top = max(df3['annotation'].value_counts())

    missing_HOM = top - df3['annotation'].value_counts()['HOM']
    missing_CON = top - df3['annotation'].value_counts()['CON']

    HOM_list = []
    CON_list = []

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
                elif row.annotation == 'CON' and len(CON_list) < missing_CON:
                    CON_list.append(list(row))
        df  = pd.DataFrame(HOM_list, columns=features.columns)
        df1 = pd.DataFrame(ANT_list, columns=features.columns)
        df2 = pd.DataFrame(CON_list, columns=features.columns)
        df3 = pd.concat([df, df3])
        df3 = pd.concat([df1, df3])
        df3 = pd.concat([df2, df3])
        dic['Y'] = df3.loc[:, ['annotation']].to_numpy()
        dic['X'] = df3.loc[:, df.columns != 'annotation'].to_numpy()
    else:
        features = df3
        while len(HOM_list) < missing_HOM:
            for row in features.itertuples(index=False):
                if row.annotation == 'HOM' and len(HOM_list) < missing_HOM:
                    HOM_list.append(list(row))
                elif row.annotation == 'CON' and len(CON_list) < missing_CON:
                    CON_list.append(list(row))
        df  = pd.DataFrame(HOM_list, columns=features.columns)
        df2 = pd.DataFrame(CON_list, columns=features.columns)
        df3 = pd.concat([df, df3])
        df3 = pd.concat([df2, df3])
        dic['Y'] = df3.loc[:, ['annotation']].to_numpy()
        dic['X'] = df3.loc[:, df.columns != 'annotation'].to_numpy()

    return dic


def set_up(classifier, num_class, list_excluded2, num_feat):
    """
    num_class = 3 (ANT, CON, HOM); 2 (CON, HOM); 4 (CON, HOM)
    classifier = 'SVM', 'MLP'
    """
    data_group  = ['all_flat']
    splits      = [41, 2, 25, 28]
    all_results = {'all_flat': {}, 'canto': {}, 'alto': {}, 'quinto': {}, 'tenor': {}, 'bass': {}}
    my_dir      = os.getcwd()

    if os.path.exists(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat)):
        shutil.rmtree(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat))
    os.mkdir(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat))

    features = pd.read_csv(my_dir + '/functionals.csv', sep='\t')

    for split in splits:
        print('Running experiments with split: ', split)
        A, B, C = get_random_split(features, split)
        ALL_dics, N_functionals = partitioning_functionals(num_class, features, A, B, C, list_excluded2)
        results = []
        for elem in data_group:
            results, all_results[elem] = run_main_function(elem, results, ALL_dics, num_class, classifier, N_functionals, all_results[elem])
            printintg_file(results, my_dir, classifier, num_class, split, elem, num_feat)

    average_results = pd.DataFrame(columns=pd.DataFrame.from_dict(all_results['all_flat']).columns)
    print('Saving results to file')

    for elem in data_group:
        print(elem)
        df_results = pd.DataFrame.from_dict(all_results[elem])
        print(df_results)
        print(df_results.mean(axis=0))
        average_results = average_results.append(df_results)
        if os.path.exists(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/' + elem + '_all_results.csv'):
            os.remove(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/' + elem + '_all_results.csv')
        df_results.to_csv(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/' + elem + '_all_results.csv', sep='\t')

    print(average_results.mean(axis=0))
    if os.path.exists(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/All_results.csv'):
        os.remove(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/All_results.csv')
    average_results.to_csv(my_dir + '/all_RESULTS_' + classifier + '_' + str(num_class) + '_' + str(num_feat) + '/All_results.csv', sep='\t')


if __name__ == '__main__':
    selected_feature = ['PS_quartile1_delta', 'PS_quartile3_delta', 'PS_mode_delta', 'PS_median_delta', 'PS_harmonic_delta', 'PS_iqr_delta', 'PS_variance_delta', 'PS_gmean_delta', 'PS_variation_delta', 'PS_skewness_delta', 'PS_kurtosis_delta', 'note_pitchPS_mean_delta', 'note_pitchPS_std_delta', 'range_PS_delta', 'min_PS_delta', 'max_PS_delta', 'PS_quartile1', 'PS_quartile3', 'PS_mode', 'PS_median', 'PS_harmonic', 'PS_iqr', 'PS_variance', 'PS_gmean', 'PS_variation', 'PS_skewness', 'PS_kurtosis', 'note_pitchPS_mean', 'note_pitchPS_std', 'range_PS', 'min_PS', 'max_PS']
    
    for classifier in ['SVM', 'MLP']:
        for num_class in [2, 3]:
            for num_feat in [163, 195]:
                if num_feat == 163:
                    list_excluded2 = selected_feature
                else:
                    list_excluded2 = []
            
                set_up(classifier, num_class, list_excluded2, num_feat)
