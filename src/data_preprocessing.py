import pandas as pd
import glob
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#this is a script for getting the data from the main surveys in the study
def get_main_survey_data(folder_path):
    my_dict = defaultdict(list)
    for filename in os.listdir(folder_path):
        if filename.startswith("result_surveys_") and str(filename).split("_")[4] != "" and str(filename).split("_")[
            4] != "tutorial" and str(filename).split("_")[5] != "2.txt":
            # print(filename)
            # my_dict["MturkID"].append(str(filename).split("_")[2])
            with open(folder_path + filename, encoding="utf-8") as f:
                # num_lines = sum(1 for line in f)
                # if num_lines == 31:
                #     print('yes')
                tempo_dict= {}
                for i, line in enumerate(f):
                    key, value = line.split(',')
                    value = value[:-1]
                    if i > 4:
                        key = key.split('_')[-3:-1]
                        key = '_'.join(key)
                    tempo_dict[key] = value
                if len(tempo_dict) == 31:
                    for key, value in tempo_dict.items():
                        my_dict[key].append(value)

    # print(my_dict)

    my_df = pd.DataFrame.from_dict(my_dict)

    return my_df

#This script get the filter question data from the post survey of the study
def get_post_survey_filter_only(folder_path):
    post_survey_dict = defaultdict(list)
    for filename in os.listdir(folder_path):

        if filename.startswith("result_surveys_") and str(filename).split("_")[5] == "2.txt" and \
                str(filename).split("_")[2] != "":
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

    post_survey_df = pd.DataFrame.from_dict(post_survey_dict)

    return post_survey_df

#This script get all data of the post survey of the study
def get_post_survey_data(folder_path):
    my_df = pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.startswith("result_surveys_") and str(filename).split("_")[5] == "2.txt" and \
                str(filename).split("_")[
                    2] != "":
            with open(folder_path + filename, encoding="utf-8") as f:
                tempo_data = {}
                for i, line in enumerate(f):
                    key, value = line.split(',')
                    value = value[:-1]
                    if i > 4:
                        key = key.split('_')[-3:-1]
                        key = '_'.join(key)

                    tempo_data[key] = tempo_data.setdefault(key, []) + [value]
                    # tempo_data[key] = value
                my_df = my_df.append(tempo_data, ignore_index=True)

    return my_df

#this script get the participants answer to the commitment question in the study
def get_commitment_data(folder_path):
    my_dict = defaultdict(list)
    for filename in os.listdir(folder_path):
        if filename.startswith("result_surveys_") and str(filename).split("_")[4] != "" and str(filename).split("_")[
            4] == "tutorial" and str(filename).split("_")[5] == "0(1).txt":
            with open(folder_path + filename, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    key, value = line.split(',')
                    value = value[:-1]
                    if i > 4:
                        key = key.split('_')[-3:-1]
                        key = '_'.join(key)
                    my_dict[key].append(value)

    my_df = pd.DataFrame.from_dict(my_dict)
    return my_df

#This script get the data from the mental wellbeing qustionnaire
def get_mental_wellbeing(folder_path, post_df):
    my_dict = defaultdict(list)
    post_array = post_df["mTurkID"].to_numpy()
    # print(post_array)
    for filename in os.listdir(folder_path):
        if filename.startswith("result_surveys_") and str(filename).split("_")[4] == "" and str(filename).split("_")[5] =="0.txt":
            # print(filename)
            # print(post_df["mTurkID"].isin([str(filename).split("_")[2]]))
            if " "+str(filename).split("_")[2] in post_array:
                # print('2', filename)
                with open(folder_path + filename, 'r') as f:
                    tempo_dict = {}
                    for i, line in enumerate(f):
                        # print(line)
                        key, value = line.split(',')
                        value = value[:-1]
                        if i > 5:
                            key = key.split('_')[-3:-1]
                            key = '_'.join(key)
                        tempo_dict[key] = value
                    if len(tempo_dict) == 17:
                        for key, value in tempo_dict.items():
                            my_dict[key].append(value)
    my_df = pd.DataFrame.from_dict(my_dict)
    return my_df

#This script get the data from the second pre-survey questionnaire aka. general positive rel and wellbeing
def get_2nd_presurvey(folder_path, post_df):
    my_dict = defaultdict(list)
    post_array = post_df["mTurkID"].to_numpy()
    # print(post_array)
    for filename in os.listdir(folder_path):
        if filename.startswith("result_surveys_") and str(filename).split("_")[4] == "" and str(filename).split("_")[5] =="0(1).txt":
            # print(filename)
            # print(post_df["mTurkID"].isin([str(filename).split("_")[2]]))
            if " "+str(filename).split("_")[2] in post_array:
                # print('2', filename)
                with open(folder_path + filename, 'r') as f:
                    tempo_dict = {}
                    for i, line in enumerate(f):
                        # print(line)
                        key, value = line.split(',')
                        value = value[:-1]
                        if i > 5:
                            key = key.split('_')[-3:-1]
                            key = '_'.join(key)
                        tempo_dict[key] = value
                    if len(tempo_dict) == 9:
                        for key, value in tempo_dict.items():
                            my_dict[key].append(value)
    my_df = pd.DataFrame.from_dict(my_dict)
    return my_df

#This function get a df (usually main survey) and categorize the data into sub dfs: gooddeed/baddeed (this is for robot) positivedeed/negativedeed (this is for scooter)
# and align/notaligned
def make_categories_main(my_df):
    gooddeed_df = my_df[my_df["drive_name"].str[2].isin(['1'])]
    baddeed_df = my_df[my_df["drive_name"].str[2].isin(['0'])]
    align_df = my_df[((my_df["drive_name"].str[3].isin(['1'])) & (my_df["2_Intention"] == '1')) | (
                (my_df["drive_name"].str[3].isin(['0'])) & (my_df["2_Intention"] == '0'))]
    notalign_df = my_df[((my_df["drive_name"].str[3].isin(['1'])) & (my_df["2_Intention"] == '0')) | (
                (my_df["drive_name"].str[3].isin(['0'])) & (my_df["2_Intention"] == '1'))]
    positive_deed_scooter_df = my_df[my_df["drive_name"].str[3].isin(['1'])]
    negative_deed_scooter_df = my_df[my_df["drive_name"].str[3].isin(['0'])]
    pos_intention_df = my_df[my_df["2_Intention"].isin(['1'])]
    neg_intention_df = my_df[my_df["2_Intention"].isin(['0'])]
    return gooddeed_df, baddeed_df, align_df, notalign_df, positive_deed_scooter_df, negative_deed_scooter_df, pos_intention_df, neg_intention_df
    # gooddeed_df.to_csv('gooddeed.csv')
    # baddeed_df.to_csv('baddeed.csv')
    # align_df.to_csv('align.csv')
    # notalign_df.to_csv('notalign.csv')
    # positive_deed_scooter_df.to_csv('pos_deed_scooter.csv')
    # negative_deed_scooter_df.to_csv('neg_deed_scooter.csv')

#this function filter out the main survey data that didn't finish the study completely
def main_survey_done_study(main_survey_df, post_survey_df):
    main_survey_done_df = main_survey_df[main_survey_df["mTurkID"].isin(post_survey_df["mTurkID"])]
    return main_survey_done_df

#this function check the color filter question (post_survey; asking about the color of the robot) and return a dfs containing the ones who passed and not passed the
# test
def color_filter_data(main_survey_df, post_survey_df):
    first_filtered_df = post_survey_df[post_survey_df["SurveyPage_Question1"] == 'Orange']
    negfirst_filtered_df = post_survey_df[post_survey_df["SurveyPage_Question1"] != 'Orange']
    return main_survey_df[main_survey_df["mTurkID"].isin(first_filtered_df["mTurkID"])], main_survey_df[main_survey_df["mTurkID"].isin(negfirst_filtered_df["mTurkID"])]

#this function check the second filter question in post survey (post_survey; asking about which mobility they were riding) and return a dfs containing the ones who
# passed and not passed the test
def scooter_filter_data(main_survey_df, post_survey_df):
    second_filtered_df = post_survey_df[(post_survey_df["SurveyPage_Question2"].str.lower().str.find('scooter') != -1)|(post_survey_df["SurveyPage_Question2"].str.lower().str.find('segway') != -1)]
    negsecond_filtered_df = post_survey_df[post_survey_df["SurveyPage_Question2"].str.lower().str.find('scooter') == -1]
    return main_survey_df[main_survey_df["mTurkID"].isin(second_filtered_df["mTurkID"])], main_survey_df[main_survey_df["mTurkID"].isin(negsecond_filtered_df["mTurkID"])]

#this function compare the main question with a +-1 of the reverse question answer
def make_reverse_compare(df, col= "1_satisfaction1"):
    condition_7 = 7 - df[col+"rx"].astype(int)
    condition_8 = 8 - df[col+"rx"].astype(int)
    condition_9 = 9 - df[col+"rx"].astype(int)
    main_con = df[col].astype(int)
    first = (main_con == condition_7)
    second = (main_con == condition_8)
    third = (main_con == condition_9)
    return (first | second | third)

#this function check the reverse questions filters in main survey (main_survey; satisfaction1 and satisfaction3 has revese question as well) and return a dfs containing
# the ones who passed and not passed the test
def rxQuestions_filter_data(main_survey_df, second_filtered_df = 'none', flag = False):

    if flag:
        filtered_main_survey_df = main_survey_df[main_survey_df["mTurkID"].isin(second_filtered_df["mTurkID"])]
        # third_filtered_df = filtered_main_survey_df[(make_reverse_compare(filtered_main_survey_df) & make_reverse_compare(filtered_main_survey_df, col = "1_satisfaction2") & make_reverse_compare(filtered_main_survey_df, col = "1_satisfaction3"))]
        #
        # negthird_filtered_df = filtered_main_survey_df[~(make_reverse_compare(filtered_main_survey_df) & make_reverse_compare(filtered_main_survey_df, col = "1_satisfaction2") & make_reverse_compare(filtered_main_survey_df, col = "1_satisfaction3"))]

        third_filtered_df = filtered_main_survey_df[(make_reverse_compare(filtered_main_survey_df) & make_reverse_compare(filtered_main_survey_df, col = "1_satisfaction3"))]

        negthird_filtered_df = filtered_main_survey_df[~(make_reverse_compare(filtered_main_survey_df) & make_reverse_compare(filtered_main_survey_df, col = "1_satisfaction3"))]


    else:
        # third_filtered_df = main_survey_df[(main_survey_df["1_satisfaction1"].astype(int) == 7- main_survey_df["1_satisfaction1rx"].astype(int)) | (main_survey_df["1_satisfaction1"].astype(int) == 8- main_survey_df["1_satisfaction1rx"].astype(int)) | (main_survey_df["1_satisfaction1"].astype(int) == 9- main_survey_df["1_satisfaction1rx"].astype(int))]
        # third_filtered_df = main_survey_df[(make_reverse_compare(main_survey_df) & make_reverse_compare(main_survey_df, col = "1_satisfaction2") & make_reverse_compare(main_survey_df, col = "1_satisfaction3"))]
        #
        # negthird_filtered_df = main_survey_df[~(make_reverse_compare(main_survey_df) & make_reverse_compare(main_survey_df, col = "1_satisfaction2") & make_reverse_compare(main_survey_df, col = "1_satisfaction3"))]

        third_filtered_df = main_survey_df[(make_reverse_compare(main_survey_df) & make_reverse_compare(main_survey_df, col = "1_satisfaction3"))]

        negthird_filtered_df = main_survey_df[~(make_reverse_compare(main_survey_df) & make_reverse_compare(main_survey_df, col = "1_satisfaction3"))]


    return third_filtered_df, negthird_filtered_df

#this function compute the total wellbeing after first interaction by averaging over the 7 factors
def mean_wellbeing_robot(df):
    df["1_total"] = df.loc[:,
                                     ["1_positive1", "1_positive2", "1_satisfaction1", "1_satisfaction2",
                                      "1_satisfaction3", "1_wellbeing", "1_trust1"]].astype('float64').mean(axis = 1)

#this function compute the total wellbeing after second interaction by averaging over the 7 factors
def mean_wellbeing_scooter(df):
    df["3_total"] = df.loc[:,
                                     ["3_positive1", "3_positive2", "3_satisfaction1", "3_satisfaction2",
                                      "3_satisfaction3", "3_wellbeing", "3_trust1"]].astype('float64').mean(axis = 1)

#this function compute the elements of wellbeing such as average of satisfaction, average of postitive relationship
def mean_elem(df, mode= 'robot'):
    if mode == 'robot':
        num = "1"
    elif mode == 'scooter':
        num = "3"
    else:
        print('please put either "robot" or "scooter" for mode')
    df[num+"_total_positive"] = df.loc[:,["1_positive1", "1_positive2"]].astype('float64').mean(axis = 1)
    df[num+"_total_satisfaction"] = df.loc[:,["1_satisfaction1", "1_satisfaction2","1_satisfaction3"]].astype('float64').mean(axis=1)

#this function compute the mean over rows for different columns
def mean_factors(df):
    new_df = df.iloc[:, 6:32].astype('float64').mean(axis=0)
    new_df["1_total"] = new_df.loc(axis=0)["1_positive1", "1_positive2", "1_satisfaction1", "1_satisfaction2",
    "1_satisfaction3", "1_wellbeing", "1_trust1"].mean(axis=0)
    new_df["3_total"] = new_df.loc(axis=0)["3_positive1", "3_positive2", "3_satisfaction1", "3_satisfaction2",
    "3_satisfaction3", "3_wellbeing", "3_trust1"].mean(axis=0)
    return new_df

#This function get two array and make the bar plot of it
def bar_plot(a, b, serieslist, barWidth = 0.25, color1='r', color2= 'g', label1='group1', label2='group2', xlabel='x', ylable='y', title = 'title', name = 'name'):

    if len(b) != 0:
        br1 = np.arange(len(a))
        br2 = [x + barWidth for x in br1]
        plt.figure(figsize=(12, 8))
        # Make the plot
        plt.bar(br1, a, color=color1, width=barWidth,
                edgecolor='grey', label=label1)
        plt.bar(br2, b, color=color2, width=barWidth,
                edgecolor='grey', label=label2)

        # Adding Xticks
        plt.xlabel(xlabel, fontweight='bold', fontsize=15)
        plt.ylabel(ylable, fontweight='bold', fontsize=15)
        plt.xticks([r + barWidth for r in range(len(a))],
                   serieslist)
        plt.title(title)
        plt.legend(loc='lower center')


        plt.savefig(name+'.png')
        plt.show()
    else:

        plt.figure(figsize=(12, 8))

        br1 = np.arange(len(a))

        # Make the plot
        plt.bar(br1, a, color=color1, width=barWidth,
                edgecolor='grey', label=label1)

        # Adding Xticks
        plt.xlabel(xlabel, fontweight='bold', fontsize=15)
        plt.ylabel(ylable, fontweight='bold', fontsize=15)
        plt.xticks([r for r in range(len(a))],
                   serieslist)
        plt.title(title)
        plt.legend(loc='lower center')



        plt.savefig(name+".png")
        plt.show()

#This function get two array and make the violin plot of it
def violin_plot(df1, df2, array1, replace_array, label1, label2, mode0 = 'value', mode1 = 'variable', mode2 = 'dataframe', title = 'title', name = 'name', order='descending'):
    df1 = df1.assign(dataframe=label1)
    df2 = df2.assign(dataframe=label2)

    long_df = pd.melt(pd.concat([df1.loc[:, array1 +["dataframe"]], df2.loc[:, array1+["dataframe"]]]),
                      id_vars=['dataframe'])

    long_df = long_df.replace(array1, replace_array)
    long_df[mode0] = pd.to_numeric(long_df[mode0])
    df_mean = long_df.groupby([mode1, mode2])[mode0].mean().reset_index()
    # df_mean.iloc[::2], df_mean.iloc[1::2] = df_mean.iloc[1::2], df_mean.iloc[::2].copy()

    if order == 'descending':
        df_mean = df_mean.sort_values(by=mode2, ascending=False)
        # print(df_mean)
    else:
        df_mean = df_mean.sort_values(by=mode2, ascending=True)

    plt.figure(figsize=(12, 8))
    sns.stripplot(x=mode1, y=mode0, data=df_mean, hue=mode2, jitter=True, dodge=True, marker='o', palette='hot_r', size=5, legend=False)
    sns.violinplot(x=mode1, y=mode0, data=long_df, hue=mode2)
    plt.legend(bbox_to_anchor=(.87, 1.06), loc='upper left', borderaxespad=0.01)
    plt.title(title)
    plt.savefig(name + ".png")
    plt.show()

#this function get all of our category of dfs and make either bar plot or violin plot (according to type) of all of our categories
def all_plot(gooddeed_df1, baddeed_df1, align_df1, notalign_df1, positive_deed_scooter_df1, negative_deed_scooter_df1, type = 'bar' , title = ''):
    title1 = 'Wellbeing Robot behavior__'+ title
    title2 = 'Wellbeing Scooter behavior__' + title
    title3 = 'Percieved robot satisfaction__' + title
    title4 = 'Trust__' + title
    name1 = 'Well-robot__' + title
    name2 = 'Well-scooter__' + title
    name3 = 'Perception__' + title
    name4 = 'Trust__'+ title
    gooddeed1 = mean_factors(gooddeed_df1)
    baddeed1 = mean_factors(baddeed_df1)
    positive1 = mean_factors(positive_deed_scooter_df1)
    negative1 = mean_factors(negative_deed_scooter_df1)
    align1 = mean_factors(align_df1)
    nalign1 = mean_factors(notalign_df1)
    array_for_robot_behavior = ["1_positive1", "1_positive2", "1_satisfaction1", "1_satisfaction2", "1_satisfaction3", "1_wellbeing", "1_trust1", "1_total"]
    label_array = ["1.positive1", "2.positive2", "3.satisfaction1", "4.satisfaction2", "5.satisfaction3", "6.wellbeing", "7.trust1", "8.total"]
    array_trust =["1_trust2", "3_trust2"]
    label_trust = ['First Interaction', 'Second Interaction']
    array_for_scooter = ["3_positive1", "3_positive2", "3_satisfaction1", "3_satisfaction2", "3_satisfaction3", "3_wellbeing", "3_trust1", "3_total"]
    array_perception = ["3_perception"]
    label_perception = ['Perception']
    if type == 'violin':
        violin_plot(gooddeed_df1, baddeed_df1, array_for_robot_behavior, label_array, 'good-deed', 'bad-deed', title = title1, name = name1)
        violin_plot(positive_deed_scooter_df1, negative_deed_scooter_df1, array_for_scooter, label_array, 'positive-deed', 'negative-deed', mode0='value', mode1='variable', mode2='dataframe', title = title2, name = name2)
        violin_plot(align_df1, notalign_df1, array_trust, label_trust, 'Aligned', 'Not-aligned', mode0='value', mode1='dataframe', mode2='variable', title = title4, name = name3, order='ascending')
        violin_plot(positive_deed_scooter_df1, negative_deed_scooter_df1, array_perception, label_perception, 'positive-deed', 'negative-deed', mode0='value', mode1='variable', mode2='dataframe', title = title3, name = name4)

    elif type == 'bar':

        bar_plot(gooddeed1[["1_positive1", "1_positive2", "1_satisfaction1", "1_satisfaction2",
                            "1_satisfaction3", "1_wellbeing", "1_trust1", "1_total"]],
                 baddeed1[["1_positive1", "1_positive2", "1_satisfaction1", "1_satisfaction2",
                           "1_satisfaction3", "1_wellbeing", "1_trust1", "1_total"]],
                 ["positive1", "positive2", "satisfaction1", "satisfaction2",
                  "satisfaction3", "wellbeing", "trust1", "total"], barWidth=0.25, color1='r', color2='g', label1='gooddeed',
                 label2='baddeed', xlabel='x', ylable='y', title = title1, name = name1)

        bar_plot(positive1[["3_positive1", "3_positive2", "3_satisfaction1", "3_satisfaction2",
                            "3_satisfaction3", "3_wellbeing", "3_trust1", "3_total"]],
                 negative1[["3_positive1", "3_positive2", "3_satisfaction1", "3_satisfaction2",
                            "3_satisfaction3", "3_wellbeing", "3_trust1", "3_total"]],
                 ["positive1", "positive2", "satisfaction1", "satisfaction2",
                  "satisfaction3", "wellbeing", "trust1", "total"], barWidth=0.25, color1='r', color2='g', label1='posdeed',
                 label2='negdeed', xlabel='x', ylable='y', title = title2, name = name2)

        bar_plot([positive1["3_perception"], negative1["3_perception"]], [],
                 ["positive-deed-perception", "negative-deed-perception"], barWidth=0.25, color1='b', color2='g',
                 label1='posdeed', label2='negdeed', xlabel='x', ylable='y', title = title3, name = name3)

        bar_plot(align1[["1_trust2", "3_trust2"]], nalign1[["1_trust2", "3_trust2"]], ["aligned", "not_aligned"],
                 barWidth=0.25, color1='b', color2='g', label1='first-interaction', label2='second-interaction', xlabel='x', ylable='y', title = title4, name = name4)

#This plot get a df and a column name and plot the distribution of the data, we can change the kind from kde to hist to have histogram plot of the distribution
def distribution_plot(df,colname,title = 'title'):
    df[colname] = pd.to_numeric(df[colname])
    df[colname].plot(kind='kde')
    # df[colname].plot(kind='hist')
    plt.title(title)
    plt.savefig(title + ".png")
    plt.show()

#this function make 2 new dfs out of main df, one has the elements of wellbeing and total wellbeing and the other has the average of the elements like total
# satisfaction, total positive etc. by changing the mode between 1 and 3 we switch from first interaction to second interaction. This function is good if we want to
# check correlation or other things specifically on each category of wellbeing instead of average
def make_corr_df(df, mode='1'):
    pos1 = df[mode+'_positive1']
    pos2 = df[mode+'_positive2']
    sat1 = df[mode+'_satisfaction1']
    sat2 = df[mode+'_satisfaction2']
    sat3 = df[mode+'_satisfaction3']
    trust1 = df[mode+'_trust1']
    well = df[mode+'_wellbeing']
    total = df[mode+'_total']
    totalpos = df[mode+'_total_positive']
    totalsat = df[mode+'_total_satisfaction']
    df_new_elem = pd.concat([pos1, pos2, sat1, sat2, sat3, trust1, well, total], axis=1)
    df_new_factor = pd.concat([totalpos, totalsat, trust1, well, total], axis=1)
    for i in df_new_elem.columns:
        df_new_elem[i] = pd.to_numeric(df_new_elem[i])
    for i in df_new_factor.columns:
        df_new_factor[i] = pd.to_numeric(df_new_factor[i])
    return df_new_elem, df_new_factor

#this funciton will get the df that is related to the postsurvey and return the percentage and number of demographic question that we specify in column
def demographic_percentage(df, colnum):
    counts = df['SurveyPage_Question'+colnum].str[0].value_counts()
    percents = counts / counts.sum() * 100

    return percents

#this function get main df survey and return a df that is based on participants. For instance when we have 300 participants who did 2 ride we will have 600 data point
# this function make it 300 to have all data of one paritcipant in a row
def make_df_participant_based(df1):
    df = df1.copy()
    # cols_to_exclude = ['name', 'mTurkID', 'subID', 'drive_index', 'time']
    col_names = []
    for i in range(df1['drive_index'].nunique()):
        for col in df1.columns:
            # if col not in ['name', 'mTurkID', 'subID', 'drive_index', 'time']:
            if col not in ['mTurkID', 'drive_index']:
                col_names.append(f'{col}_{i}')

        # pivot the dataframe to have participants as rows and drive_index as columns
    pivoted_df = df.pivot(index='mTurkID', columns='drive_index')
    pivoted_df.columns = [f'{col[0]}_{col[1]}' for col in pivoted_df.columns]

    # rename the columns as per your requirement
    df = pivoted_df.reindex(columns=col_names)
    # pivoted_df.columns = col_names

    # reset the index of the dataframe
    df = df.reset_index()

    new_col_names = ['mTurkID']
    for i in range(df1['drive_index'].nunique()):
        for col in df1.columns:
            if col not in ['mTurkID', 'drive_index']:
                new_col_names.append(f'{col}_{i}')
    df.columns = new_col_names

    return df
#this function is to check the policy condition over the data and return a df whom the study had the condition of the optimal policy
def categorize_CloseLoop(df):

    lowwell_df = df[df['1_total'] <= 2.08]
    highwell_df = df[df['1_total'] > 2.08]
    low = lowwell_df[lowwell_df["drive_name"].str[3].isin(['0'])]
    high = highwell_df[highwell_df["drive_name"].str[3].isin(['1'])]
    our_condition_df = pd.concat([low, high], axis=0)

    return our_condition_df