
import pandas as pd

import glob

from collections import defaultdict

import os
folder_path = './Mturk/data-old-pilot/'
my_dict = defaultdict(list)
for filename in os.listdir("./Mturk/data/"):
    if filename.startswith("result_surveys_") and str(filename).split("_")[4] != "" and str(filename).split("_")[4] != "tutorial" and str(filename).split("_")[5] != "2.txt":
        # print(filename)
        my_dict["MturkID"].append(str(filename).split("_")[2])
        with open(folder_path + filename, encoding="utf-8") as f:

            for i, line in enumerate(f):
                key, value = line.split(',')
                value = value[:-1]
                if i > 4:
                    key = key.split('_')[-4:-1]
                    key = '_'.join(key)
                my_dict[key].append(value)
# print(my_dict)


my_df = pd.DataFrame.from_dict(my_dict)
my_df.to_csv('result.csv')


gooddeed_df = my_df[my_df["drive_name"].str[2].isin(['1'])]
baddeed_df = my_df[my_df["drive_name"].str[2].isin(['0'])]
align_df = my_df[((my_df["drive_name"].str[3].isin(['1'])) & (my_df["Intersection_2_Intention"] == '1')) | ((my_df["drive_name"].str[3].isin(['0'])) & (my_df["Intersection_2_Intention"] == '2'))]
notalign_df = my_df[((my_df["drive_name"].str[3].isin(['1'])) & (my_df["Intersection_2_Intention"] == '2')) | ((my_df["drive_name"].str[3].isin(['0'])) & (my_df["Intersection_2_Intention"] == '1'))]
positive_deed_scooter_df = my_df[my_df["drive_name"].str[3].isin(['1'])]
negative_deed_scooter_df = my_df[my_df["drive_name"].str[3].isin(['0'])]

# gooddeed_df["total_wellbeing"] = gooddeed_df.loc[:, ["1_positive1", "1_positive2", "1_satisfaction1", "1_satisfaction2", "1_satisfaction3", "1_wellbeing", "1_trust1"]]


gooddeed_df.to_csv('gooddeed.csv')
baddeed_df.to_csv('baddeed.csv')
align_df.to_csv('align.csv')
notalign_df.to_csv('notalign.csv')
positive_deed_scooter_df.to_csv('pos_deed_scooter.csv')
negative_deed_scooter_df.to_csv('neg_deed_scooter.csv')