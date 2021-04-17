import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_feature_groups():
    # CATEGORICAL
    music_text = ['mel_ratio', 'syl_num', 'mel_num']

    # CONTINUOUS
    beat          = ['beat_quartile1', 'beat_quartile3', 'beat_mode', 'beat_median', 'beat_harmonic', 'beat_iqr', 'beat_variance', 'beat_gmean', 'beat_variation', 'beat_skewness', 'beat_kurtosis', 'beat_max', 'beat_min', 'beat_mean', 'beat_std', 'beat_diff']
    pitch         = ['PS_quartile1', 'PS_quartile3', 'PS_mode', 'PS_median', 'PS_harmonic', 'PS_iqr', 'PS_variance', 'PS_gmean', 'PS_variation', 'PS_skewness', 'PS_kurtosis', 'note_pitchPS_mean', 'note_pitchPS_std', 'range_PS', 'min_PS', 'max_PS']
    interval_up   = ['interval_posit_quartile1', 'interval_posit_quartile3', 'interval_posit_mode', 'interval_posit_median', 'interval_posit_harmonic', 'interval_posit_iqr', 'interval_posit_variance', 'interval_posit_gmean', 'interval_posit_variation', 'interval_posit_skewness', 'interval_posit_kurtosis', 'interval_mean_posit', 'interval_std_posit', 'interval_max_posit', 'interval_min_posit', 'interval_diff_posit']
    interval_down = ['interval_negat_quartile1', 'interval_negat_quartile3', 'interval_negat_mode', 'interval_negat_median', 'interval_negat_harmonic', 'interval_negat_iqr', 'interval_negat_variance', 'interval_negat_gmean', 'interval_negat_variation', 'interval_negat_skewness', 'interval_negat_kurtosis', 'interval_mean_negat', 'interval_min_negat', 'interval_max_negat', 'interval_diff_negat', 'interval_std_negat']
    rhythm        = ['rhythm_quartile1', 'rhythm_quartile3', 'rhythm_mode', 'rhythm_median', 'rhythm_harmonic', 'rhythm_iqr', 'rhythm_variance', 'rhythm_gmean', 'rhythm_variation', 'rhythm_skewness', 'rhythm_kurtosis', 'rhythm_mean', 'rhythm_std', 'rhythm_min', 'rhythm_max', 'rhythm_diff']
    offset        = ['offset_quartile1', 'offset_quartile3', 'offset_mode', 'offset_median', 'offset_harmonic', 'offset_iqr', 'offset_variance', 'offset_gmean', 'offset_variation', 'offset_skewness', 'offset_kurtosis', 'offset_mean', 'offset_std', 'offset_max', 'offset_min', 'offset_diff']

    # DELTAS
    beat_delta          = ['beat_quartile1_delta', 'beat_quartile3_delta', 'beat_mode_delta', 'beat_median_delta', 'beat_harmonic_delta', 'beat_iqr_delta', 'beat_variance_delta', 'beat_gmean_delta', 'beat_variation_delta', 'beat_skewness_delta', 'beat_kurtosis_delta', 'beat_max_delta', 'beat_min_delta', 'beat_mean_delta', 'beat_std_delta', 'beat_diff_delta']
    pitch_delta         = ['PS_quartile1_delta', 'PS_quartile3_delta', 'PS_mode_delta', 'PS_median_delta', 'PS_harmonic_delta', 'PS_iqr_delta', 'PS_variance_delta', 'PS_gmean_delta', 'PS_variation_delta', 'PS_skewness_delta', 'PS_kurtosis_delta', 'note_pitchPS_mean_delta', 'note_pitchPS_std_delta', 'range_PS_delta', 'min_PS_delta', 'max_PS_delta']
    interval_up_delta   = ['interval_posit_quartile1_delta', 'interval_posit_quartile3_delta', 'interval_posit_mode_delta', 'interval_posit_median_delta', 'interval_posit_harmonic_delta', 'interval_posit_iqr_delta', 'interval_posit_variance_delta', 'interval_posit_gmean_delta', 'interval_posit_variation_delta', 'interval_posit_skewness_delta', 'interval_posit_kurtosis_delta', 'interval_mean_posit_delta', 'interval_std_posit_delta', 'interval_max_posit_delta', 'interval_min_posit_delta', 'interval_diff_posit_delta']
    interval_down_delta = ['interval_negat_quartile1_delta', 'interval_negat_quartile3_delta', 'interval_negat_mode_delta', 'interval_negat_median_delta', 'interval_negat_harmonic_delta', 'interval_negat_iqr_delta', 'interval_negat_variance_delta', 'interval_negat_gmean_delta', 'interval_negat_variation_delta', 'interval_negat_skewness_delta', 'interval_negat_kurtosis_delta', 'interval_mean_negat_delta', 'interval_min_negat_delta', 'interval_max_negat_delta', 'interval_diff_negat_delta', 'interval_std_negat_delta']
    rhythm_delta        = ['rhythm_quartile1_delta', 'rhythm_quartile3_delta', 'rhythm_mode_delta', 'rhythm_median_delta', 'rhythm_harmonic_delta', 'rhythm_iqr_delta', 'rhythm_variance_delta', 'rhythm_gmean_delta', 'rhythm_variation_delta', 'rhythm_skewness_delta', 'rhythm_kurtosis_delta', 'rhythm_mean_delta', 'rhythm_std_delta', 'rhythm_min_delta', 'rhythm_max_delta', 'rhythm_diff_delta']
    offset_delta        = ['offset_quartile1_delta', 'offset_quartile3_delta', 'offset_mode_delta', 'offset_median_delta', 'offset_harmonic_delta', 'offset_iqr_delta', 'offset_variance_delta', 'offset_gmean_delta', 'offset_variation_delta', 'offset_skewness_delta', 'offset_kurtosis_delta', 'offset_mean_delta', 'offset_std_delta', 'offset_max_delta', 'offset_min_delta', 'offset_diff_delta']

    pitch_CON    = pitch + pitch_delta
    rhythm_CON   = rhythm + rhythm_delta
    interval_CON = interval_down + interval_up + interval_down_delta + interval_up_delta
    attack_CON   = beat + offset + beat_delta + offset_delta

    # GROUPS FOR PCA
    # vertical dimension (freq)=note; diagonal dimension (frep + time)=interval; horizontal dimension (time)=rhythm
    feature_dic = {'note_ALL': pitch_CON, 'interval_ALL': interval_CON + music_text, 'rhythm_ALL': rhythm_CON + attack_CON}

    return feature_dic


def run_PCA(df, principalDf, feature_dic):
    y = df.loc[:, ['annotation']].values

    for elem in feature_dic:
        x   = df.loc[:, feature_dic[elem]].values
        x   = StandardScaler().fit_transform(x)
        pca = PCA(n_components=1)

        principalComponents = pca.fit_transform(x)
        print(elem)
        print(pca.explained_variance_ratio_)
        principalDf.insert(loc=0, column=elem, value=principalComponents.flatten().tolist(), allow_duplicates=True)

    principalDf.insert(loc=0, column='Annotation', value=y)
    print(principalDf)

    return principalDf


def set_up(part):
    my_dir      = os.getcwd()
    df          = pd.read_csv(my_dir + '/functionals.csv', sep='\t')
    df          = df.loc[df['part_name'] == part]
    principalDf = pd.DataFrame()
    return df, principalDf, my_dir


if __name__ == '__main__':
    parts = ['all_flat', 'Bass', 'Tenor', 'Quinto', 'Canto', 'Alto']

    for part in parts:
        print(part)
        df, principalDf, my_dir = set_up(part)
        feature_dic = get_feature_groups()
        principalDf = run_PCA(df, principalDf, feature_dic)
        if os.path.exists(my_dir + '/PCA_' + part + '.csv'):
            os.remove(my_dir + '/PCA_' + part + '.csv')
        principalDf.to_csv(my_dir + '/PCA_' + part + '.csv', sep=';')
