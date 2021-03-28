#!/usr/bin/env python
# -- coding: utf-8 --
import os
import glob
import shutil
import statistics
from scipy import stats
import pandas as pd
import numpy as np
import math
from music21 import *
import warnings
warnings.filterwarnings('ignore')


def reshape_LLDs(my_dir):
    if os.path.exists(my_dir + '/LLD_Deltas_all2'):
        shutil.rmtree(my_dir + '/LLD_Deltas_all2')

    os.mkdir(my_dir + '/LLD_Deltas_all2')
    min_rhythm = 0.25  # 1/16 note in music21 (the shortest note in the corpus)

    for elem in os.listdir(my_dir + '/LLD_Deltas_all'):
        print('Processing: ', elem)
        os.mkdir(my_dir + '/LLD_Deltas_all2/' + elem)
        for part in os.listdir(my_dir + '/LLD_Deltas_all/' + elem):
            if part != 'all_flat.csv':
                mad = pd.read_csv(my_dir + '/LLD_Deltas_all/' + elem + '/' + part, sep='\t', engine='python')
                df = pd.DataFrame(columns=mad.columns.tolist()[1:])
                index = 0
                for cell in mad['rhythm']:
                    rep_row = round(cell/min_rhythm)
                    for num in range(rep_row):
                        df = df.append(mad.iloc[index])
                    index = index + 1
                print(len(df))
                df.to_csv(my_dir + '/LLD_Deltas_all2/' + elem + '/' + part, sep='\t')


def get_music21(madPart_content, annotations, LLD, composer, part_name, part_clef):
    bar_num_list      = []
    beat_num_list     = []
    hum_pos_list      = []
    note_pitchPS_list = []
    note_name_list    = []
    text_list         = []
    plain_notes       = []
    rhythm_list       = []
    offset_list       = []

    for h in madPart_content:
        bar_num_list.append(h.measureNumber)
        beat_num_list.append(round(float(h.beat), 2))
        hum_pos_list.append(h.humdrumPosition+1)
        rhythm_list.append(round(float(h.duration.quarterLength), 2))
        offset_list.append(round(float(h.offset), 2))
        if isinstance(h, note.Note):
            plain_notes.append(h)
            note_pitchPS_list.append(h.pitch.midi)
            note_name_list.append(h.name)
            if h.lyrics:
                text_list.append(h.lyric)
            else:
                text_list.append('-')
        else:
            plain_notes.append(h)
            note_pitchPS_list.append('-')
            note_name_list.append('REST')
            text_list.append('-')

    counter       = 0
    max_counter   = len(plain_notes) - 1
    interval_list = []

    for x in plain_notes:
        if counter < max_counter:
            n1 = plain_notes[counter]
            n2 = plain_notes[counter + 1]
            if isinstance(n1, note.Note) and isinstance(n2, note.Note):
                i = interval.Interval(noteStart=n1, noteEnd=n2)
                interval_list.append((i.semitones, i.directedName))
            else:
                interval_list.append(('-', '-'))
            counter = counter + 1

    interval_list.insert(0, (0, 0))
    annotations_list = []
    target_annot     = ['CON', 'HOM', 'ANT']

    for key in annotations:
        for hum_row in annotations[key]:
            if any(annot in hum_row[0] for annot in target_annot):
                if 'CON' in hum_row[0] and 'HOM' not in hum_row[0] and 'ANT' not in hum_row[0]:
                    local_annot = 'CON'
                elif 'HOM' in hum_row[0] and 'CON' not in hum_row[0] and 'ANT' not in hum_row[0]:
                    local_annot = 'HOM'
                elif 'ANT' in hum_row[0] and 'HOM' not in hum_row[0] and 'CON' not in hum_row[0]:
                    local_annot = 'ANT'
                else:
                    local_annot = 'MIX'
                annotations_list.append((local_annot, hum_row[1], hum_row[2]))
            elif '>' in hum_row[0]:
                annotations_list.append((local_annot, hum_row[1], hum_row[2]))
                local_annot = 'NONE'
            else:
                annotations_list.append((local_annot, hum_row[1], hum_row[2]))

    annotations_list_part = []
    for L in annotations_list:
        if L[1] in hum_pos_list:
            annotations_list_part.append((L[0], L[2]))

    melismas_list = []
    for index, name in enumerate(note_name_list):
        if name != 'REST':
            if index == len(text_list) - 1:
                if text_list[index] != '-':
                    melismas_list.append(1)
                else:
                    melismas_list.append(-1)
            elif text_list[index] != '-' and text_list[index + 1] != '-':
                melismas_list.append(1)
            else:
                if note_name_list[index + 1] != 'REST':
                    melismas_list.append(-1)
                else:
                    melismas_list.append(1)
        else:
            melismas_list.append(0)

    text_list = [accent.replace('&agrave', 'à') for accent in text_list]
    text_list = [accent.replace('&egrave', 'è') for accent in text_list]
    text_list = [accent.replace('&igrave', 'ì') for accent in text_list]
    text_list = [accent.replace('&ograve', 'ò') for accent in text_list]
    text_list = [accent.replace('&ugrave', 'ù') for accent in text_list]
    text_list = [accent.replace('&aacute', 'á') for accent in text_list]
    text_list = [accent.replace('&eacute', 'é') for accent in text_list]
    text_list = [accent.replace('&iacute', 'í') for accent in text_list]
    text_list = [accent.replace('&oacute', 'ó') for accent in text_list]
    text_list = [accent.replace('&uacute', 'ú') for accent in text_list]

    # get delta coefficients
    beat_num_list_delta     = [0]
    note_pitchPS_list_delta = [0]
    interval_list_delta     = [0]
    rhythm_list_delta       = [0]
    offset_list_delta       = [0]
    beat_num_list_delta     = get_deltas(beat_num_list, beat_num_list_delta)
    note_pitchPS_list_delta = get_deltas(note_pitchPS_list, note_pitchPS_list_delta)
    interval_list_delta     = get_deltas(list(list(zip(*interval_list))[0]), interval_list_delta)
    rhythm_list_delta       = get_deltas(rhythm_list, rhythm_list_delta)
    offset_list_delta       = get_deltas(offset_list, offset_list_delta)

    for index, annot in enumerate(annotations_list_part):
        LLD['composer'].append(composer)
        LLD['part_name'].append(part_name)
        LLD['part_clef'].append(part_clef)
        LLD['bar_num'].append(bar_num_list[index])
        LLD['beat_num'].append(beat_num_list[index])
        LLD['hum_pos'].append(hum_pos_list[index])
        LLD['note_pitchPS'].append(note_pitchPS_list[index])
        LLD['note_name'].append(note_name_list[index])
        LLD['text'].append(text_list[index])
        LLD['text_mus'].append(melismas_list[index])
        LLD['interval_num'].append(interval_list[index][0])
        LLD['interval_name'].append(interval_list[index][1])
        LLD['rhythm'].append(rhythm_list[index])
        LLD['offset'].append(offset_list[index])
        LLD['beat_num_delta'].append(beat_num_list_delta[index])
        LLD['note_pitchPS_delta'].append(note_pitchPS_list_delta[index])
        LLD['interval_num_delta'].append(interval_list_delta[index])
        LLD['rhythm_delta'].append(rhythm_list_delta[index])
        LLD['offset_delta'].append(offset_list_delta[index])
        LLD['annotation'].append(annot[0])
        LLD['annotation_ID'].append(annot[1])

    return LLD


def get_annotations(file_name, path):
    f           = open(path + '/' + file_name + '.krn', 'rt', encoding='utf8')
    annotations = {}
    my_key      = str()
    annotation_counter = 0

    for index, line in enumerate(f):
        if (line.find('=') > -1) and (line.find('==') < 0) and (line.find('RDF') < 0) and (line.find('=||') < 0) and (line.find('!') < 0):
            a, b = line.split('\t', 1)
            if a == '=1-':
                my_key = '1'
            else:
                n, my_key = a.split('=')
            annotations[my_key] = []
        else:
            if my_key:
                if line.find('*') < 0 and line.find('=') < 0:
                    c, d = line.rsplit('\t', 1)
                    if d != '.\n':
                        if '<CON' in d or '<HOM' in d or '<ANT' in d:
                            annotation_counter = annotation_counter + 1
                        annotations[my_key].append((d.rstrip(), index + 1, annotation_counter))
                    else:
                        annotations[my_key].append((d.rstrip(), index + 1, annotation_counter))
    f.close()

    return annotations


def get_deltas(LLD_list, delta_list):
    for index, elem in enumerate(LLD_list):
        if index + 1 < len(LLD_list):
            if isinstance(LLD_list[index+1], str) or isinstance(LLD_list[index], str):
                delta_list.append('-')
            else:
                delta_list.append(LLD_list[index+1] - LLD_list[index])
    return delta_list


def get_tied_notes(my_element, my_list):
    if isinstance(my_element, note.Note):
        if (my_element.tie is not None) and (my_element.tie.type == 'stop' or my_element.tie.type == 'continue'):
            back_index = -1
            while isinstance(my_list[back_index], tuple):
                back_index -= 1
            my_list[back_index].duration.quarterLength += my_element.duration.quarterLength
        else:
            my_list.append(my_element)
    elif isinstance(my_element, note.Rest):
        my_list.append(my_element)


def extract_LLD(my_dir):
    LLD         = {'composer': [], 'part_name': [], 'part_clef': [], 'bar_num': [], 'beat_num': [], 'hum_pos': [], 'note_pitchPS': [], 'note_name': [], 'text': [], 'text_mus': [], 'interval_num': [], 'interval_name': [], 'rhythm': [], 'offset': [], 'beat_num_delta': [], 'note_pitchPS_delta': [], 'interval_num_delta': [], 'rhythm_delta': [], 'offset_delta': [], 'annotation': [], 'annotation_ID': []}
    localCorpus = corpus.corpora.LocalCorpus()

    if os.path.exists(my_dir + '/LLDs_Deltas_all.csv'):
        os.remove(my_dir + '/LLDs_Deltas_all.csv')

    localCorpus.addPath(my_dir + '/corpus')
    path = my_dir + '/corpus'

    for krn_file in glob.glob(os.path.join(path, '*.krn')):
        file_name = os.path.basename(krn_file[0:-4])
        print('Processing ', file_name)
        composer, other = file_name.split('_', 1)

        annotations = get_annotations(file_name, path)
        s           = converter.parse(path + '/' + file_name + '.krn')
        NumParts    = len(s.parts)
        count       = NumParts - 1

        while count > -1:
            if count == 4:
                part_name = 'Bass'
            elif count == 3:
                part_name = 'Tenor'
            elif count == 2:
                part_name = 'Quinto'
            elif count == 1:
                part_name = 'Alto'
            else:
                part_name = 'Canto'

            madrigal_part    = s.parts[count]
            madPart_elements = madrigal_part.elements
            madPart_content  = []

            for n in madPart_elements:
                if isinstance(n, stream.Measure):
                    for x in n.elements:
                        if isinstance(x, clef.Clef):
                            part_clef = x.name
                        else:
                            get_tied_notes(x, madPart_content)

            LLD   = get_music21(madPart_content, annotations, LLD, composer, part_name, part_clef)
            count = count - 1

        # get flattened data
        s         = s.flat
        part_name = 'all_flat'
        part_clef = '_'
        madPart_elements = s.elements
        madPart_content  = []

        for n in madPart_elements:
            get_tied_notes(n, madPart_content)
        LLD = get_music21(madPart_content, annotations, LLD, composer, part_name, part_clef)

    df = pd.DataFrame.from_dict(LLD)
    print(df)
    df.to_csv(my_dir + '/LLDs_Deltas_all.csv', sep='\t')


def convert_nan_inf(my_var):
    if (math.isnan(my_var)) or (my_var == np.inf):
        my_var = 0
    return my_var


def fix_empty_lists(var_list):
    if len(var_list) > 1:
        var_max       = max(var_list)
        var_min       = min(var_list)
        var_mean      = statistics.mean(var_list)
        var_quartile1 = np.percentile(var_list, 25)
        var_quartile3 = np.percentile(var_list, 75)
        var_std       = statistics.stdev(var_list)
        var_diff      = var_max - var_min
        var_gmean     = stats.gmean(var_list)
        var_gmean     = convert_nan_inf(var_gmean)
        var_harmonic  = statistics.harmonic_mean([abs(ele) for ele in var_list])
        for num in var_list:
            if num < 0:
                var_harmonic = var_harmonic*-1
                break
        var_median    = statistics.median(var_list)
        var_mode      = stats.mode(var_list)[0][0]
        var_kurtosis  = stats.kurtosis(var_list)
        var_skewness  = stats.skew(var_list)
        var_variation = stats.variation(var_list)
        var_variation = convert_nan_inf(var_variation)
        var_variance  = statistics.variance(var_list)
        var_iqr       = stats.iqr(var_list)
        var_iqr       = convert_nan_inf(var_iqr)
    elif len(var_list) == 1:
        var_max       = var_list[0]
        var_min       = var_list[0]
        var_mean      = var_list[0]
        var_std       = 0
        var_diff      = var_max - var_min
        var_kurtosis  = 0
        var_skewness  = 0
        var_variation = 0
        var_gmean     = 0
        var_variance  = 0
        var_iqr       = 0
        var_harmonic  = 0
        var_median    = 0
        var_mode      = 0
        var_quartile3 = 0
        var_quartile1 = 0
    else:
        var_max       = 0
        var_min       = 0
        var_mean      = 0
        var_std       = 0
        var_diff      = var_max - var_min
        var_kurtosis  = 0
        var_skewness  = 0
        var_variation = 0
        var_gmean     = 0
        var_variance  = 0
        var_iqr       = 0
        var_harmonic  = 0
        var_median    = 0
        var_mode      = 0
        var_quartile3 = 0
        var_quartile1 = 0
    return var_quartile1, var_quartile3, var_mode, var_median, var_harmonic, var_iqr, var_variance, var_gmean, var_variation, var_skewness, var_kurtosis, var_max, var_min, var_mean, var_std, var_diff


def get_ratio(nom, den):
    if den > 0:
        ratio = nom / den
    else:
        ratio = 0
    return ratio


def extract_functionals(my_dir):
    functionals = {'composer': [], 'part_name': [], 'part_clef': [], 'mel_ratio': [], 'syl_num': [], 'mel_num': [], 'beat_quartile1_delta': [], 'beat_quartile3_delta': [], 'beat_mode_delta': [], 'beat_median_delta': [], 'beat_harmonic_delta': [], 'beat_iqr_delta': [], 'beat_variance_delta': [], 'beat_gmean_delta': [], 'beat_variation_delta': [], 'beat_skewness_delta': [], 'beat_kurtosis_delta': [], 'PS_quartile1_delta': [], 'PS_quartile3_delta': [], 'PS_mode_delta': [], 'PS_median_delta': [], 'PS_harmonic_delta': [], 'PS_iqr_delta': [], 'PS_variance_delta': [], 'PS_gmean_delta': [], 'PS_variation_delta': [], 'PS_skewness_delta': [], 'PS_kurtosis_delta': [], 'interval_negat_quartile1_delta': [], 'interval_negat_quartile3_delta': [], 'interval_negat_mode_delta': [], 'interval_negat_median_delta': [], 'interval_negat_harmonic_delta': [], 'interval_negat_iqr_delta': [], 'interval_negat_variance_delta': [], 'interval_negat_gmean_delta': [], 'interval_negat_variation_delta': [], 'interval_negat_skewness_delta': [], 'interval_negat_kurtosis_delta': [], 'interval_posit_quartile1_delta': [], 'interval_posit_quartile3_delta': [], 'interval_posit_mode_delta': [], 'interval_posit_median_delta': [], 'interval_posit_harmonic_delta': [], 'interval_posit_iqr_delta': [], 'interval_posit_variance_delta': [], 'interval_posit_gmean_delta': [], 'interval_posit_variation_delta': [], 'interval_posit_skewness_delta': [], 'interval_posit_kurtosis_delta': [], 'offset_quartile1_delta': [], 'offset_quartile3_delta': [], 'offset_mode_delta': [], 'offset_median_delta': [], 'offset_harmonic_delta': [], 'offset_iqr_delta': [], 'offset_variance_delta': [], 'offset_gmean_delta': [], 'offset_variation_delta': [], 'offset_skewness_delta': [], 'offset_kurtosis_delta': [], 'rhythm_quartile1_delta': [], 'rhythm_quartile3_delta': [], 'rhythm_mode_delta': [], 'rhythm_median_delta': [], 'rhythm_harmonic_delta': [], 'rhythm_iqr_delta': [], 'rhythm_variance_delta': [], 'rhythm_gmean_delta': [], 'rhythm_variation_delta': [], 'rhythm_skewness_delta': [], 'rhythm_kurtosis_delta': [], 'beat_quartile1': [], 'beat_quartile3': [], 'beat_mode': [], 'beat_median': [], 'beat_harmonic': [], 'beat_iqr': [], 'beat_variance': [], 'beat_gmean': [], 'beat_variation': [], 'beat_skewness': [], 'beat_kurtosis': [], 'PS_quartile1': [], 'PS_quartile3': [], 'PS_mode': [], 'PS_median': [], 'PS_harmonic': [], 'PS_iqr': [], 'PS_variance': [], 'PS_gmean': [], 'PS_variation': [], 'PS_skewness': [], 'PS_kurtosis': [], 'interval_negat_quartile1': [], 'interval_negat_quartile3': [], 'interval_negat_mode': [], 'interval_negat_median': [], 'interval_negat_harmonic': [], 'interval_negat_iqr': [], 'interval_negat_variance': [], 'interval_negat_gmean': [], 'interval_negat_variation': [], 'interval_negat_skewness': [], 'interval_negat_kurtosis': [], 'interval_posit_quartile1': [], 'interval_posit_quartile3': [], 'interval_posit_mode': [], 'interval_posit_median': [], 'interval_posit_harmonic': [], 'interval_posit_iqr': [], 'interval_posit_variance': [], 'interval_posit_gmean': [], 'interval_posit_variation': [], 'interval_posit_skewness': [], 'interval_posit_kurtosis': [], 'offset_quartile1': [], 'offset_quartile3': [], 'offset_mode': [], 'offset_median': [], 'offset_harmonic': [], 'offset_iqr': [], 'offset_variance': [], 'offset_gmean': [], 'offset_variation': [], 'offset_skewness': [], 'offset_kurtosis': [], 'rhythm_quartile1': [], 'rhythm_quartile3': [], 'rhythm_mode': [], 'rhythm_median': [], 'rhythm_harmonic': [], 'rhythm_iqr': [], 'rhythm_variance': [], 'rhythm_gmean': [], 'rhythm_variation': [], 'rhythm_skewness': [], 'rhythm_kurtosis': [], 'beat_max': [], 'beat_min': [], 'beat_mean': [], 'beat_std': [], 'beat_diff': [], 'beat_max_delta': [], 'beat_min_delta': [], 'beat_mean_delta': [], 'beat_std_delta': [], 'beat_diff_delta': [], 'note_pitchPS_mean': [], 'note_pitchPS_std': [], 'note_pitchPS_mean_delta': [], 'note_pitchPS_std_delta': [], 'range_PS': [], 'min_PS': [], 'max_PS': [], 'range_PS_delta': [], 'min_PS_delta': [], 'max_PS_delta': [], 'interval_mean_posit': [], 'interval_std_posit': [], 'interval_max_posit': [], 'interval_min_posit': [], 'interval_diff_posit': [], 'interval_mean_negat': [], 'interval_min_negat': [], 'interval_max_negat': [], 'interval_diff_negat': [], 'interval_std_negat': [], 'interval_mean_posit_delta': [], 'interval_std_posit_delta': [], 'interval_max_posit_delta': [], 'interval_min_posit_delta': [], 'interval_diff_posit_delta': [], 'interval_mean_negat_delta': [], 'interval_min_negat_delta': [], 'interval_max_negat_delta': [], 'interval_diff_negat_delta': [], 'interval_std_negat_delta': [], 'rhythm_mean': [], 'rhythm_std': [], 'rhythm_min': [], 'rhythm_max': [], 'rhythm_diff': [], 'offset_mean': [], 'offset_std': [], 'offset_max': [], 'offset_min': [], 'offset_diff': [], 'rhythm_mean_delta': [], 'rhythm_std_delta': [], 'rhythm_min_delta': [], 'rhythm_max_delta': [], 'rhythm_diff_delta': [], 'offset_mean_delta': [], 'offset_std_delta': [], 'offset_max_delta': [], 'offset_min_delta': [], 'offset_diff_delta': [], 'annotation': [], 'annotation_ID': []}
    if os.path.exists(my_dir + '/functionals.csv'):
        os.remove(my_dir + '/functionals.csv')
    LLDs = pd.read_csv('LLDs_Deltas_all.csv', sep='\t')

    # split data by composer
    composers      = LLDs['composer'].unique().tolist()
    Composers_Dict = {elem: pd.DataFrame for elem in composers}
    for composer_key in Composers_Dict.keys():
        Composers_Dict[composer_key] = LLDs[:][LLDs.composer == composer_key]

    # split data (within composer) by annotation ID
    for dataframe in Composers_Dict:
        print('Processing ', dataframe)
        PARTS = Composers_Dict[dataframe]['part_name'].unique().tolist()
        PART_Dict = {elem: pd.DataFrame for elem in PARTS}
        for part_key in PART_Dict.keys():
            PART_Dict[part_key] = Composers_Dict[dataframe][:][Composers_Dict[dataframe].part_name == part_key]
        for part_dataframe in PART_Dict:
            print('Extracting functionals for ', part_dataframe)
            IDs = PART_Dict[part_dataframe]['annotation_ID'].unique().tolist()
            ID_Dict = {elem: pd.DataFrame for elem in IDs}

            # collect funtionals for each madrigalism
            for ID_key in ID_Dict.keys():
                ID_Dict[ID_key] = PART_Dict[part_dataframe][:][PART_Dict[part_dataframe].annotation_ID == ID_key]
                composer_f      = ID_Dict[ID_key].iloc[0]['composer']
                part_name_f     = ID_Dict[ID_key].iloc[0]['part_name']
                part_clef_f     = ID_Dict[ID_key].iloc[0]['part_clef']
                beat_quartile1, beat_quartile3, beat_mode, beat_median, beat_harmonic, beat_iqr, beat_variance, beat_gmean, beat_variation, beat_skewness, beat_kurtosis, beat_max, beat_min, beat_mean, beat_std, beat_diff = fix_empty_lists(ID_Dict[ID_key]['beat_num'].tolist())
                beat_quartile1_delta, beat_quartile3_delta, beat_mode_delta, beat_median_delta, beat_harmonic_delta, beat_iqr_delta, beat_variance_delta, beat_gmean_delta, beat_variation_delta, beat_skewness_delta, beat_kurtosis_delta, beat_max_delta, beat_min_delta, beat_mean_delta, beat_std_delta, beat_diff_delta = fix_empty_lists(ID_Dict[ID_key]['beat_num_delta'].tolist())
                PS_list = []
                for elem in ID_Dict[ID_key]['note_pitchPS']:
                    if elem != '-':
                        PS_list.append(int(elem))
                PS_list_delta = []
                for elem in ID_Dict[ID_key]['note_pitchPS_delta']:
                    if elem != '-':
                        PS_list_delta.append(int(elem))
                PS_quartile1, PS_quartile3, PS_mode, PS_median, PS_harmonic, PS_iqr, PS_variance, PS_gmean, PS_variation, PS_skewness, PS_kurtosis, max_PS, min_PS, note_pitchPS_mean, note_pitchPS_std, range_PS = fix_empty_lists(PS_list)
                PS_quartile1_delta, PS_quartile3_delta, PS_mode_delta, PS_median_delta, PS_harmonic_delta, PS_iqr_delta, PS_variance_delta, PS_gmean_delta, PS_variation_delta, PS_skewness_delta, PS_kurtosis_delta, max_PS_delta, min_PS_delta, note_pitchPS_mean_delta, note_pitchPS_std_delta, range_PS_delta = fix_empty_lists(PS_list_delta)
                notes_list = []
                rests_list = []
                for elem in ID_Dict[ID_key]['note_name']:
                    if elem != 'REST':
                        notes_list.append(elem)
                    else:
                        rests_list.append(elem)
                mel_list  = []
                syl_list  = []
                none_list = []
                for elem in ID_Dict[ID_key]['text_mus']:
                    if elem == 1:
                        syl_list.append(elem)
                    elif elem == -1:
                        mel_list.append(elem)
                    else:
                        none_list.append(elem)
                mel_num       = len(mel_list)
                syl_num       = len(syl_list)
                mel_ratio     = get_ratio(syl_num, mel_num)
                interval_list = []
                for elem in ID_Dict[ID_key]['interval_num']:
                    if elem != '-':
                        interval_list.append(int(elem))
                interval_list_delta = []
                for elem in ID_Dict[ID_key]['interval_num_delta']:
                    if elem != '-':
                        interval_list_delta.append(int(elem))
                positive_list = []
                negative_list = []
                unisone_list  = []
                for elem in interval_list:
                    if elem > 0:
                        positive_list.append(elem)
                    elif elem == 0:
                        unisone_list.append(elem)
                    else:
                        negative_list.append(elem)
                positive_list_delta = []
                negative_list_delta = []
                for elem in interval_list_delta:
                    if elem > 0:
                        positive_list_delta.append(elem)
                    elif elem < 0:
                        negative_list_delta.append(elem)
                interval_posit_quartile1, interval_posit_quartile3, interval_posit_mode, interval_posit_median, interval_posit_harmonic, interval_posit_iqr, interval_posit_variance, interval_posit_gmean, interval_posit_variation, interval_posit_skewness, interval_posit_kurtosis, interval_max_posit, interval_min_posit, interval_mean_posit, interval_std_posit, interval_diff_posit = fix_empty_lists(positive_list)
                interval_negat_quartile1, interval_negat_quartile3, interval_negat_mode, interval_negat_median, interval_negat_harmonic, interval_negat_iqr, interval_negat_variance, interval_negat_gmean, interval_negat_variation, interval_negat_skewness, interval_negat_kurtosis, interval_max_negat, interval_min_negat, interval_mean_negat, interval_std_negat, interval_diff_negat = fix_empty_lists(negative_list)
                interval_posit_quartile1_delta, interval_posit_quartile3_delta, interval_posit_mode_delta, interval_posit_median_delta, interval_posit_harmonic_delta, interval_posit_iqr_delta, interval_posit_variance_delta, interval_posit_gmean_delta, interval_posit_variation_delta, interval_posit_skewness_delta, interval_posit_kurtosis_delta, interval_max_posit_delta, interval_min_posit_delta, interval_mean_posit_delta, interval_std_posit_delta, interval_diff_posit_delta = fix_empty_lists(positive_list_delta)
                interval_negat_quartile1_delta, interval_negat_quartile3_delta, interval_negat_mode_delta, interval_negat_median_delta, interval_negat_harmonic_delta, interval_negat_iqr_delta, interval_negat_variance_delta, interval_negat_gmean_delta, interval_negat_variation_delta, interval_negat_skewness_delta, interval_negat_kurtosis_delta, interval_max_negat_delta, interval_min_negat_delta, interval_mean_negat_delta, interval_std_negat_delta, interval_diff_negat_delta = fix_empty_lists(negative_list_delta)
                rhythm_quartile1, rhythm_quartile3, rhythm_mode, rhythm_median, rhythm_harmonic, rhythm_iqr, rhythm_variance, rhythm_gmean, rhythm_variation, rhythm_skewness, rhythm_kurtosis, rhythm_max, rhythm_min, rhythm_mean, rhythm_std, rhythm_diff = fix_empty_lists(ID_Dict[ID_key]['rhythm'].tolist())
                offset_quartile1, offset_quartile3, offset_mode, offset_median, offset_harmonic, offset_iqr, offset_variance, offset_gmean, offset_variation, offset_skewness, offset_kurtosis, offset_max, offset_min, offset_mean, offset_std, offset_diff = fix_empty_lists(ID_Dict[ID_key]['offset'].tolist())
                rhythm_quartile1_delta, rhythm_quartile3_delta, rhythm_mode_delta, rhythm_median_delta, rhythm_harmonic_delta, rhythm_iqr_delta, rhythm_variance_delta, rhythm_gmean_delta, rhythm_variation_delta, rhythm_skewness_delta, rhythm_kurtosis_delta, rhythm_max_delta, rhythm_min_delta, rhythm_mean_delta, rhythm_std_delta, rhythm_diff_delta = fix_empty_lists(ID_Dict[ID_key]['rhythm_delta'].tolist())
                offset_quartile1_delta, offset_quartile3_delta, offset_mode_delta, offset_median_delta, offset_harmonic_delta, offset_iqr_delta, offset_variance_delta, offset_gmean_delta, offset_variation_delta, offset_skewness_delta, offset_kurtosis_delta, offset_max_delta, offset_min_delta, offset_mean_delta, offset_std_delta, offset_diff_delta = fix_empty_lists(ID_Dict[ID_key]['offset_delta'].tolist())
                annotation_f    = ID_Dict[ID_key].iloc[0]['annotation']
                annotation_ID_f = ID_Dict[ID_key].iloc[0]['annotation_ID']

                # fill out the funtionals dataframe
                functionals['composer'].append(composer_f)
                functionals['part_name'].append(part_name_f)
                functionals['part_clef'].append(part_clef_f)
                functionals['beat_mean'].append(beat_mean)
                functionals['beat_max'].append(beat_max)
                functionals['beat_min'].append(beat_min)
                functionals['beat_std'].append(beat_std)
                functionals['beat_diff'].append(beat_diff)
                functionals['beat_mean_delta'].append(beat_mean_delta)
                functionals['beat_max_delta'].append(beat_max_delta)
                functionals['beat_min_delta'].append(beat_min_delta)
                functionals['beat_std_delta'].append(beat_std_delta)
                functionals['beat_diff_delta'].append(beat_diff_delta)
                functionals['note_pitchPS_mean'].append(note_pitchPS_mean)
                functionals['note_pitchPS_std'].append(note_pitchPS_std)
                functionals['note_pitchPS_mean_delta'].append(note_pitchPS_mean_delta)
                functionals['note_pitchPS_std_delta'].append(note_pitchPS_std_delta)
                functionals['range_PS'].append(range_PS)
                functionals['min_PS'].append(min_PS)
                functionals['max_PS'].append(max_PS)
                functionals['range_PS_delta'].append(range_PS_delta)
                functionals['min_PS_delta'].append(min_PS_delta)
                functionals['max_PS_delta'].append(max_PS_delta)
                functionals['interval_mean_posit'].append(interval_mean_posit)
                functionals['interval_std_posit'].append(interval_std_posit)
                functionals['interval_max_posit'].append(interval_max_posit)
                functionals['interval_min_posit'].append(interval_min_posit)
                functionals['interval_diff_posit'].append(interval_diff_posit)
                functionals['interval_mean_negat'].append(interval_mean_negat)
                functionals['interval_min_negat'].append(interval_min_negat)
                functionals['interval_max_negat'].append(interval_max_negat)
                functionals['interval_diff_negat'].append(interval_diff_negat)
                functionals['interval_std_negat'].append(interval_std_negat)
                functionals['interval_mean_posit_delta'].append(interval_mean_posit_delta)
                functionals['interval_std_posit_delta'].append(interval_std_posit_delta)
                functionals['interval_max_posit_delta'].append(interval_max_posit_delta)
                functionals['interval_min_posit_delta'].append(interval_min_posit_delta)
                functionals['interval_diff_posit_delta'].append(interval_diff_posit_delta)
                functionals['interval_mean_negat_delta'].append(interval_mean_negat_delta)
                functionals['interval_min_negat_delta'].append(interval_min_negat_delta)
                functionals['interval_max_negat_delta'].append(interval_max_negat_delta)
                functionals['interval_diff_negat_delta'].append(interval_diff_negat_delta)
                functionals['interval_std_negat_delta'].append(interval_std_negat_delta)
                functionals['rhythm_mean'].append(rhythm_mean)
                functionals['rhythm_std'].append(rhythm_std)
                functionals['rhythm_min'].append(rhythm_min)
                functionals['rhythm_max'].append(rhythm_max)
                functionals['rhythm_diff'].append(rhythm_diff)
                functionals['offset_mean'].append(offset_mean)
                functionals['offset_std'].append(offset_std)
                functionals['offset_max'].append(offset_max)
                functionals['offset_min'].append(offset_min)
                functionals['offset_diff'].append(offset_diff)
                functionals['rhythm_mean_delta'].append(rhythm_mean_delta)
                functionals['rhythm_std_delta'].append(rhythm_std_delta)
                functionals['rhythm_min_delta'].append(rhythm_min_delta)
                functionals['rhythm_max_delta'].append(rhythm_max_delta)
                functionals['rhythm_diff_delta'].append(rhythm_diff_delta)
                functionals['offset_mean_delta'].append(offset_mean_delta)
                functionals['offset_std_delta'].append(offset_std_delta)
                functionals['offset_max_delta'].append(offset_max_delta)
                functionals['offset_min_delta'].append(offset_min_delta)
                functionals['offset_diff_delta'].append(offset_diff_delta)
                functionals['mel_ratio'].append(mel_ratio)
                functionals['syl_num'].append(syl_num)
                functionals['mel_num'].append(mel_num)
                functionals['annotation'].append(annotation_f)
                functionals['annotation_ID'].append(annotation_ID_f)
                functionals['beat_quartile1_delta'].append(beat_quartile1_delta)
                functionals['beat_quartile3_delta'].append(beat_quartile3_delta)
                functionals['beat_mode_delta'].append(beat_mode_delta)
                functionals['beat_median_delta'].append(beat_median_delta)
                functionals['beat_harmonic_delta'].append(beat_harmonic_delta)
                functionals['beat_iqr_delta'].append(beat_iqr_delta)
                functionals['beat_variance_delta'].append(beat_variance_delta)
                functionals['beat_gmean_delta'].append(beat_gmean_delta)
                functionals['beat_variation_delta'].append(beat_variation_delta)
                functionals['beat_skewness_delta'].append(beat_skewness_delta)
                functionals['beat_kurtosis_delta'].append(beat_kurtosis_delta)
                functionals['PS_quartile1_delta'].append(PS_quartile1_delta)
                functionals['PS_quartile3_delta'].append(PS_quartile3_delta)
                functionals['PS_mode_delta'].append(PS_mode_delta)
                functionals['PS_median_delta'].append(PS_median_delta)
                functionals['PS_harmonic_delta'].append(PS_harmonic_delta)
                functionals['PS_iqr_delta'].append(PS_iqr_delta)
                functionals['PS_variance_delta'].append(PS_variance_delta)
                functionals['PS_gmean_delta'].append(PS_gmean_delta)
                functionals['PS_variation_delta'].append(PS_variation_delta)
                functionals['PS_skewness_delta'].append(PS_skewness_delta)
                functionals['PS_kurtosis_delta'].append(PS_kurtosis_delta)
                functionals['interval_negat_quartile1_delta'].append(interval_negat_quartile1_delta)
                functionals['interval_negat_quartile3_delta'].append(interval_negat_quartile3_delta)
                functionals['interval_negat_mode_delta'].append(interval_negat_mode_delta)
                functionals['interval_negat_median_delta'].append(interval_negat_median_delta)
                functionals['interval_negat_harmonic_delta'].append(interval_negat_harmonic_delta)
                functionals['interval_negat_iqr_delta'].append(interval_negat_iqr_delta)
                functionals['interval_negat_variance_delta'].append(interval_negat_variance_delta)
                functionals['interval_negat_gmean_delta'].append(interval_negat_gmean_delta)
                functionals['interval_negat_variation_delta'].append(interval_negat_variation_delta)
                functionals['interval_negat_skewness_delta'].append(interval_negat_skewness_delta)
                functionals['interval_negat_kurtosis_delta'].append(interval_negat_kurtosis_delta)
                functionals['interval_posit_quartile1_delta'].append(interval_posit_quartile1_delta)
                functionals['interval_posit_quartile3_delta'].append(interval_posit_quartile3_delta)
                functionals['interval_posit_mode_delta'].append(interval_posit_mode_delta)
                functionals['interval_posit_median_delta'].append(interval_posit_median_delta)
                functionals['interval_posit_harmonic_delta'].append(interval_posit_harmonic_delta)
                functionals['interval_posit_iqr_delta'].append(interval_posit_iqr_delta)
                functionals['interval_posit_variance_delta'].append(interval_posit_variance_delta)
                functionals['interval_posit_gmean_delta'].append(interval_posit_gmean_delta)
                functionals['interval_posit_variation_delta'].append(interval_posit_variation_delta)
                functionals['interval_posit_skewness_delta'].append(interval_posit_skewness_delta)
                functionals['interval_posit_kurtosis_delta'].append(interval_posit_kurtosis_delta)
                functionals['offset_quartile1_delta'].append(offset_quartile1_delta)
                functionals['offset_quartile3_delta'].append(offset_quartile3_delta)
                functionals['offset_mode_delta'].append(offset_mode_delta)
                functionals['offset_median_delta'].append(offset_median_delta)
                functionals['offset_harmonic_delta'].append(offset_harmonic_delta)
                functionals['offset_iqr_delta'].append(offset_iqr_delta)
                functionals['offset_variance_delta'].append(offset_variance_delta)
                functionals['offset_gmean_delta'].append(offset_gmean_delta)
                functionals['offset_variation_delta'].append(offset_variation_delta)
                functionals['offset_skewness_delta'].append(offset_skewness_delta)
                functionals['offset_kurtosis_delta'].append(offset_kurtosis_delta)
                functionals['rhythm_quartile1_delta'].append(rhythm_quartile1_delta)
                functionals['rhythm_quartile3_delta'].append(rhythm_quartile3_delta)
                functionals['rhythm_mode_delta'].append(rhythm_mode_delta)
                functionals['rhythm_median_delta'].append(rhythm_median_delta)
                functionals['rhythm_harmonic_delta'].append(rhythm_harmonic_delta)
                functionals['rhythm_iqr_delta'].append(rhythm_iqr_delta)
                functionals['rhythm_variance_delta'].append(rhythm_variance_delta)
                functionals['rhythm_gmean_delta'].append(rhythm_gmean_delta)
                functionals['rhythm_variation_delta'].append(rhythm_variation_delta)
                functionals['rhythm_skewness_delta'].append(rhythm_skewness_delta)
                functionals['rhythm_kurtosis_delta'].append(rhythm_kurtosis_delta)
                functionals['beat_quartile1'].append(beat_quartile1)
                functionals['beat_quartile3'].append(beat_quartile3)
                functionals['beat_mode'].append(beat_mode)
                functionals['beat_median'].append(beat_median)
                functionals['beat_harmonic'].append(beat_harmonic)
                functionals['beat_iqr'].append(beat_iqr)
                functionals['beat_variance'].append(beat_variance)
                functionals['beat_gmean'].append(beat_gmean)
                functionals['beat_variation'].append(beat_variation)
                functionals['beat_skewness'].append(beat_skewness)
                functionals['beat_kurtosis'].append(beat_kurtosis)
                functionals['PS_quartile1'].append(PS_quartile1)
                functionals['PS_quartile3'].append(PS_quartile3)
                functionals['PS_mode'].append(PS_mode)
                functionals['PS_median'].append(PS_median)
                functionals['PS_harmonic'].append(PS_harmonic)
                functionals['PS_iqr'].append(PS_iqr)
                functionals['PS_variance'].append(PS_variance)
                functionals['PS_gmean'].append(PS_gmean)
                functionals['PS_variation'].append(PS_variation)
                functionals['PS_skewness'].append(PS_skewness)
                functionals['PS_kurtosis'].append(PS_kurtosis)
                functionals['interval_negat_quartile1'].append(interval_negat_quartile1)
                functionals['interval_negat_quartile3'].append(interval_negat_quartile3)
                functionals['interval_negat_mode'].append(interval_negat_mode)
                functionals['interval_negat_median'].append(interval_negat_median)
                functionals['interval_negat_harmonic'].append(interval_negat_harmonic)
                functionals['interval_negat_iqr'].append(interval_negat_iqr)
                functionals['interval_negat_variance'].append(interval_negat_variance)
                functionals['interval_negat_gmean'].append(interval_negat_gmean)
                functionals['interval_negat_variation'].append(interval_negat_variation)
                functionals['interval_negat_skewness'].append(interval_negat_skewness)
                functionals['interval_negat_kurtosis'].append(interval_negat_kurtosis)
                functionals['interval_posit_quartile1'].append(interval_posit_quartile1)
                functionals['interval_posit_quartile3'].append(interval_posit_quartile3)
                functionals['interval_posit_mode'].append(interval_posit_mode)
                functionals['interval_posit_median'].append(interval_posit_median)
                functionals['interval_posit_harmonic'].append(interval_posit_harmonic)
                functionals['interval_posit_iqr'].append(interval_posit_iqr)
                functionals['interval_posit_variance'].append(interval_posit_variance)
                functionals['interval_posit_gmean'].append(interval_posit_gmean)
                functionals['interval_posit_variation'].append(interval_posit_variation)
                functionals['interval_posit_skewness'].append(interval_posit_skewness)
                functionals['interval_posit_kurtosis'].append(interval_posit_kurtosis)
                functionals['offset_quartile3'].append(offset_quartile3)
                functionals['offset_quartile1'].append(offset_quartile1)
                functionals['offset_mode'].append(offset_mode)
                functionals['offset_median'].append(offset_median)
                functionals['offset_harmonic'].append(offset_harmonic)
                functionals['offset_iqr'].append(offset_iqr)
                functionals['offset_variance'].append(offset_variance)
                functionals['offset_gmean'].append(offset_gmean)
                functionals['offset_variation'].append(offset_variation)
                functionals['offset_skewness'].append(offset_skewness)
                functionals['offset_kurtosis'].append(offset_kurtosis)
                functionals['rhythm_quartile1'].append(rhythm_quartile1)
                functionals['rhythm_quartile3'].append(rhythm_quartile3)
                functionals['rhythm_mode'].append(rhythm_mode)
                functionals['rhythm_median'].append(rhythm_median)
                functionals['rhythm_harmonic'].append(rhythm_harmonic)
                functionals['rhythm_iqr'].append(rhythm_iqr)
                functionals['rhythm_variance'].append(rhythm_variance)
                functionals['rhythm_gmean'].append(rhythm_gmean)
                functionals['rhythm_variation'].append(rhythm_variation)
                functionals['rhythm_skewness'].append(rhythm_skewness)
                functionals['rhythm_kurtosis'].append(rhythm_kurtosis)

    df = pd.DataFrame.from_dict(functionals)
    print(df)
    df.to_csv(os.getcwd() + '/functionals.csv', sep='\t')


def process_LLDs(my_dir):
    if os.path.exists(my_dir + '/LLD_Deltas_all'):
        shutil.rmtree(my_dir + '/LLD_Deltas_all')
    os.mkdir(my_dir + '/LLD_Deltas_all')
    LLDs = pd.read_csv('LLDs_Deltas_all.csv', sep='\t')

    # split data by composer
    composers      = LLDs['composer'].unique().tolist()
    order_folders  = []
    f              = open(my_dir + "/folders_order.txt", "w")
    Composers_Dict = {elem: pd.DataFrame for elem in composers}
    for composer_key in Composers_Dict:
        Composers_Dict[composer_key] = LLDs[:][LLDs.composer == composer_key]

    # split data (within composer) by annotation ID
    for dataframe in Composers_Dict:
        print('Splitting by ', dataframe)
        IDs     = Composers_Dict[dataframe]['annotation_ID'].unique().tolist()
        ID_Dict = {elem: pd.DataFrame for elem in IDs}
        for ID_key in ID_Dict:
            ID_Dict[ID_key] = Composers_Dict[dataframe][:][Composers_Dict[dataframe].annotation_ID == ID_key]
        for dataframe_ID in ID_Dict:
            annotation = ID_Dict[dataframe_ID].iloc[0]['annotation']
            if os.path.exists(my_dir + '/LLD_Deltas_all/' + str(dataframe_ID) + '_' + dataframe + '_' + annotation):
                os.remove(my_dir + '/LLD_Deltas_all/' + str(dataframe_ID) + '_' + dataframe + '_' + annotation)
            os.mkdir(my_dir + '/LLD_Deltas_all/' + str(dataframe_ID) + '_' + dataframe + '_' + annotation)
            order_folders.append(str(dataframe_ID) + '_' + dataframe + '_' + annotation)
            print('Getting parts in madrigalism ', dataframe_ID)
            parts      = ID_Dict[dataframe_ID]['part_name'].unique().tolist()
            parts_Dict = {elem: pd.DataFrame for elem in parts}
            for part_key in parts_Dict:
                parts_Dict[part_key] = ID_Dict[dataframe_ID][:][ID_Dict[dataframe_ID].part_name == part_key]
                for elem in parts_Dict[part_key].columns.values.tolist():
                    parts_Dict[part_key][elem] = parts_Dict[part_key][elem].replace('-', 0)
                parts_Dict[part_key].to_csv(my_dir + '/LLD_Deltas_all/' + str(dataframe_ID) + '_' + dataframe + '_' + annotation + '/' + part_key + '.csv', sep='\t')
    for line in order_folders:
        print(line, file=f)
    f.close()


def get_features(my_dir):
    if not os.path.exists(my_dir + '/LLDs_Deltas_all.csv'):
        extract_LLD(my_dir)
    if not os.path.exists(my_dir + '/functionals.csv'):
        extract_functionals(my_dir)
    if not os.path.exists(my_dir + '/LLD_Deltas_all'):
        process_LLDs(my_dir)


if __name__ == '__main__':
    my_dir = os.getcwd()
    get_features(my_dir)
    reshape_LLDs(my_dir)
    if os.path.exists(my_dir + '/LLDs_Deltas_all.csv'):
        os.remove(my_dir + '/LLDs_Deltas_all.csv')
    if os.path.exists(my_dir + '/LLD_Deltas_all'):
        shutil.rmtree(my_dir + '/LLD_Deltas_all')
