import pandas as pd

import glob

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
folder_path = './Mturk/data/'
post_survey_dict = defaultdict(list)
main_survey_dict = defaultdict(list)
for filename in os.listdir(folder_path):
    if filename.startswith("result_surveys_") and str(filename).split("_")[5] == "2.txt" and str(filename).split("_")[2] != "":
        # print(filename)
        # my_dict["MturkID"].append(str(filename).split("_")[2])
        with open(folder_path + filename, encoding="utf-8") as f:

            for i, line in enumerate(f):
                if i < 8:

                    key, value = line.split(',')
                    value = value[:-1]
                    if i > 5:
                        key = key.split('_')[-3:-1]
                        key = '_'.join(key)
                    post_survey_dict[key].append(value)

    if filename.startswith("result_surveys_") and str(filename).split("_")[4] != "" and str(filename).split("_")[4] != "tutorial" and str(filename).split("_")[5] != "2.txt" and str(filename).split("_")[2] != "":
        # print(filename)
        with open(folder_path + filename, encoding="utf-8") as f:
            tempo_dict = {}
            for i, line in enumerate(f):
                key, value = line.split(',')
                value = value[:-1]
                if i > 5:
                    key = key.split('_')[-3:-1]
                    key = '_'.join(key)
                tempo_dict[key] = value
            if len(tempo_dict) == 31:
                for key, value in tempo_dict.items():
                    main_survey_dict[key].append(value)
# print(my_dict)


post_survey_df = pd.DataFrame.from_dict(post_survey_dict)
main_survey_df = pd.DataFrame.from_dict(main_survey_dict)
first_filtered_df = post_survey_df[post_survey_df["SurveyPage_Question1"] == 'Orange']

# second_filtered_df = first_filtered_df[first_filtered_df["SurveyPage_Question2"].str.lower().str.find('scooter') != -1]
second_filtered_df = post_survey_df[post_survey_df["SurveyPage_Question2"].str.lower().str.find('scooter') != -1]
# filtered_main_survey_df = main_survey_df[main_survey_df["mTurkID"].isin(second_filtered_df["mTurkID"])]
# third_filtered_df = filtered_main_survey_df[(filtered_main_survey_df["1_satisfaction1"].astype(int) == 7- filtered_main_survey_df["1_satisfaction1rx"].astype(int)) | (filtered_main_survey_df["1_satisfaction1"].astype(int) == 8- filtered_main_survey_df["1_satisfaction1rx"].astype(int)) | (filtered_main_survey_df["1_satisfaction1"].astype(int) == 9- filtered_main_survey_df["1_satisfaction1rx"].astype(int))]

# second_filtered_df = post_survey_df[post_survey_df["SurveyPage_Question2"].str.lower().str.find('scooter') != -1]
third_filtered_df = main_survey_df[(main_survey_df["1_satisfaction1"].astype(int) == 7- main_survey_df["1_satisfaction1rx"].astype(int)) | (main_survey_df["1_satisfaction1"].astype(int) == 8- main_survey_df["1_satisfaction1rx"].astype(int)) | (main_survey_df["1_satisfaction1"].astype(int) == 9- main_survey_df["1_satisfaction1rx"].astype(int))]



# print(post_survey_df.shape[0])
# print(main_survey_df.shape[0]/2)
# # print(post_survey_df["SurveyPage_Question1"])
# print(first_filtered_df.shape[0])
# print(second_filtered_df.shape[0])
# print(third_filtered_df.shape[0]/2)
# # print(third_filtered_df.shape[0]/2, post_survey_df.shape[0], third_filtered_df.shape[0]/ (post_survey_df.shape[0]*2))
# print(third_filtered_df)
# print(main_survey_df)

# my_df.to_csv('result.csv')

gooddeed_df = third_filtered_df[third_filtered_df["drive_name"].str[2].isin(['1'])]
baddeed_df = third_filtered_df[third_filtered_df["drive_name"].str[2].isin(['0'])]
# align_df = third_filtered_df[third_filtered_df["drive_name"].str[3] == third_filtered_df["2_Intention"].str]
# # print(third_filtered_df["drive_name"].str[3], third_filtered_df["2_Intention"])
align_df = third_filtered_df[((third_filtered_df["drive_name"].str[3].isin(['1'])) & (third_filtered_df["2_Intention"] == '1')) | ((third_filtered_df["drive_name"].str[3].isin(['0'])) & ((third_filtered_df["2_Intention"] == '0') | (third_filtered_df["2_Intention"] == '2')))]
notalign_df = third_filtered_df[(((third_filtered_df["drive_name"].str[3].isin(['1'])) & ((third_filtered_df["2_Intention"] == '0') | (third_filtered_df["2_Intention"] == '2'))) | ((third_filtered_df["drive_name"].str[3].isin(['0'])) & (third_filtered_df["2_Intention"] == '1')))]
positive_deed_scooter_df = third_filtered_df[third_filtered_df["drive_name"].str[3].isin(['1'])]
negative_deed_scooter_df = third_filtered_df[third_filtered_df["drive_name"].str[3].isin(['0'])]
# gooddeed_df.to_csv('gooddeed1.csv')
# baddeed_df.to_csv('baddeed1.csv')
# align_df.to_csv('align1.csv')
# notalign_df.to_csv('notalign1.csv')
# positive_deed_scooter_df.to_csv('pos_deed_scooter1.csv')
# negative_deed_scooter_df.to_csv('neg_deed_scooter1.csv')

# gooddeed_df["total_wellbeing"] = gooddeed_df.loc[:, ["1_positive1", "1_positive2", "1_satisfaction1", "1_satisfaction2", "1_satisfaction3", "1_wellbeing", "1_trust1"]].astype('float64').mean(axis=1)
# # gooddeed_df["total_wellbeing"] = df_mean.mean(axis = 1)
# print(gooddeed_df)
# print(gooddeed_df.iloc[:, 6:31].astype('float64').mean(axis=0))
# new_df = gooddeed_df.mean(axis = 0)
# print(new_df)


# gooddeed_df = gooddeed_df.assign(dataframe='gooddeed')
# baddeed_df = baddeed_df.assign(dataframe='baddeed')
#
# long_df = pd.melt(pd.concat([gooddeed_df.loc[:,["1_positive1", "1_positive2", "1_satisfaction1", "1_satisfaction2", "1_satisfaction3", "1_wellbeing", "1_trust1", "dataframe"]], baddeed_df.loc[:,["1_positive1", "1_positive2", "1_satisfaction1", "1_satisfaction2", "1_satisfaction3", "1_wellbeing", "1_trust1", "dataframe"]]]), id_vars=['dataframe'])
#
# long_df['value'] = pd.to_numeric(long_df['value'])
# plt.figure(figsize=(10,6))
# sns.violinplot(x="variable", y="value", data=long_df, hue="dataframe")
# plt.legend(bbox_to_anchor=(.9, 1.06), loc='upper left', borderaxespad=0.01)

align_df = align_df.assign(dataframe='Aligned')
notalign_df = notalign_df.assign(dataframe='Not-aligned')

long_df = pd.melt(pd.concat([align_df.loc[:,["1_trust2", "3_trust2", "dataframe"]], notalign_df.loc[: ,["1_trust2", "3_trust2", "dataframe"]]]), id_vars=['dataframe'])
long_df = long_df.replace(['1_trust2','3_trust2'],['First Interaction', 'Second Interaction'])
print(long_df)
long_df['value'] = pd.to_numeric(long_df['value'])
plt.figure(figsize=(10,6))
sns.violinplot(x="dataframe", y="value", data=long_df, hue="variable")
plt.legend(bbox_to_anchor=(.88, 1.06), loc='upper left', borderaxespad=0.01)

# sns.violinplot(data=long_df, col = 'variable', )
# long_df['variable'] = pd.to_numeric(long_df['variable'])
# g = sns.FacetGrid(long_df, col='variable', col_wrap=4, hue='dataframe')
# g.map(sns.violinplot, 'value', orient='v', split=True)
plt.show()

# plt.show()