from stat_tests import *
from data_preprocessing import *



# folder_path = './Mturk/data/'
folder_path = './Prolific/data_all/'
pd.set_option('display.max_columns', 600)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_colwidth', -1)

##############
#Getting data
##############
main_survey_df = get_main_survey_data(folder_path)
post_survey_df = get_post_survey_filter_only(folder_path)
main_survey_done_df = main_survey_done_study(main_survey_df,post_survey_df)
post_survey_df_nofilter = get_post_survey_data(folder_path)
mental_wellbeing_df = get_mental_wellbeing(folder_path, post_survey_df)
mental_wellbeing_df['total_well'] = mental_wellbeing_df.iloc[:, 6:15].astype('float64').mean(axis=1)
presurvey_2nd_df = get_2nd_presurvey(folder_path, post_survey_df)
# print(presurvey_2nd_df)
# print(post_survey_df_nofilter)
# print(mental_wellbeing_df)
mean_wellbeing_robot(main_survey_df)
mean_wellbeing_scooter(main_survey_df)
mean_wellbeing_robot(main_survey_done_df)
mean_wellbeing_scooter(main_survey_done_df)
mean_elem(main_survey_done_df)
mean_elem(main_survey_done_df, mode='scooter')

#getting the data for filter questions
first_filter_df, negfirst_filter_df = color_filter_data(main_survey_df, post_survey_df)
second_filter_df, negsecond_filter_df = scooter_filter_data(main_survey_df, post_survey_df)
third_filter_df, negthird_filter_df= rxQuestions_filter_data(main_survey_df)
first_second_df, negfirst_second_df = scooter_filter_data(first_filter_df, post_survey_df)
second_third_df, negsecond_third_df = rxQuestions_filter_data(second_filter_df)
first_third_df, negfirst_third_df = rxQuestions_filter_data(first_second_df)
all_filter_df, negall_filter_df = rxQuestions_filter_data(first_second_df)
commit_df = get_commitment_data(folder_path)
# print(main_survey_done_df)

#getting the data that has each participant data in a row
df_with_turkid = make_df_participant_based(main_survey_done_df)

#Making csv files from the dataframes
main_survey_done_df.to_csv('main_survey_done.csv')
post_survey_df_nofilter.to_csv('post_survey.csv')
mental_wellbeing_df.to_csv('mental_wellbeing_presurvey.csv')
presurvey_2nd_df.to_csv('2nd_presurvey.csv')
# print(df_with_turkid)

###################
#Rates for filters
###################
print('commitment rate: ', "%",commit_df[commit_df['SurveyPage_Question1']=='yes'].shape[0]/commit_df.shape[0]*100)
print('main_survey', main_survey_df.shape[0]/2, '\n', 'post_survey', post_survey_df.shape[0], '\n', 'main_survey_done', main_survey_done_df.shape[0]/2, '\n', 'first_filter', first_filter_df.shape[0]/2, negfirst_filter_df.shape[0]/2, '\n', 'second_filter', second_filter_df.shape[0]/2, negsecond_filter_df.shape[0]/2, '\n', 'third_filter', third_filter_df.shape[0]/2, negthird_filter_df.shape[0]/2, '\n', 'first_second_filter', first_second_df.shape[0]/2, '\n', 'first_third_filter', first_third_df.shape[0]/2, '\n', 'second_third_filter', second_third_df.shape[0]/2, '\n', 'all_filter', all_filter_df.shape[0]/2)
# print('third_filter', third_filter_df.shape[0]/2, negthird_filter_df.shape[0]/2,'\n','third_filter2', third_filter_df2.shape[0]/2, negthird_filter_df2.shape[0]/2,'\n','third_filter3', third_filter_df3.shape[0]/2, negthird_filter_df3.shape[0]/2,)

# print(main_survey_done_df)

#checking the number of each category
gooddeed_df, baddeed_df, align_df, notalign_df, positive_deed_scooter_df, negative_deed_scooter_df, pos_intention_df, neg_intention_df = make_categories_main(
    main_survey_done_df)
print(align_df.shape[0], notalign_df.shape[0], '\n', gooddeed_df.shape[0], baddeed_df.shape[0], '\n', positive_deed_scooter_df.shape[0],
      negative_deed_scooter_df.shape[0], '\n', pos_intention_df.shape[0], neg_intention_df.shape[0])
#########
#Plots
#########
# all_plot(gooddeed_df, baddeed_df, align_df, notalign_df, positive_deed_scooter_df, negative_deed_scooter_df, type = 'violin', title= 'no_filter')
# all_plot(gooddeed_df, baddeed_df, align_df, notalign_df, align_df, notalign_df, type = 'violin', title= 'no_filter_align')

#categorizing the data of whom passed third filter (reversed questions)
gooddeed_df3, baddeed_df3, align_df3, notalign_df3, positive_deed_scooter_df3, negative_deed_scooter_df3, pos_intention_df3, neg_intention_df3 = make_categories_main(third_filter_df)
# all_plot(gooddeed_df3, baddeed_df3, align_df3, notalign_df3, positive_deed_scooter_df3, negative_deed_scooter_df3, type = 'violin', title= 'third_filter')

#ploting distribution of different data
# distribution_plot(main_survey_done_df,'1_total')
# distribution_plot(main_survey_done_df,'3_satisfaction3', title = 'satisfaction3')
# distribution_plot(positive_deed_scooter_df,'3_total')
# distribution_plot(negative_deed_scooter_df,'3_total')
# print(main_survey_done_df)


#############
#Stat tests
# #############
first_well = '1_total'
second_well = '3_total'
# elem = 'positive2'
# first_well = '1_'+elem
# second_well = '3_'+elem
print('\n','T_tests')
t_test(gooddeed_df[first_well], baddeed_df[first_well], mode='greater')
t_test(gooddeed_df['1_satisfaction3'], baddeed_df['1_satisfaction3'], mode='greater')
# t_test(gooddeed_df['1_total_satisfaction'], baddeed_df['1_total_satisfaction'], mode='greater')
t_test(gooddeed_df['2_Intention'], baddeed_df['2_Intention'])
t_test(positive_deed_scooter_df[second_well], negative_deed_scooter_df[second_well])
t_test(positive_deed_scooter_df['3_perception'], negative_deed_scooter_df['3_perception'], mode='greater')
t_test(positive_deed_scooter_df['3_trust2'], negative_deed_scooter_df['3_trust2'], mode='greater')
t_test(align_df[second_well], notalign_df[second_well],mode='greater')
t_test(align_df['3_trust2'], notalign_df['3_trust2'], mode='greater')
elem_intent = '1_total'
# elem_intent = '1_total_positive'
t_test(pos_intention_df[elem_intent], neg_intention_df[elem_intent], mode='greater')
print('\n','Mann-Whitney tests')
mann_whitney(gooddeed_df[first_well], baddeed_df[first_well], mode='greater')
mann_whitney(gooddeed_df['1_satisfaction3'], baddeed_df['1_satisfaction3'], mode='greater')
mann_whitney(gooddeed_df['2_Intention'], baddeed_df['2_Intention'])
mann_whitney(positive_deed_scooter_df[second_well], negative_deed_scooter_df[second_well])
mann_whitney(positive_deed_scooter_df['3_trust2'], negative_deed_scooter_df['3_trust2'], mode='greater')
mann_whitney(align_df[second_well], notalign_df[second_well],mode='greater')
mann_whitney(align_df['3_trust2'], notalign_df['3_trust2'], mode='greater')
mann_whitney(pos_intention_df[elem_intent], neg_intention_df[elem_intent], mode='greater')
print('\n','Correlation tests')
pearson_corr(main_survey_done_df['3_trust2'], main_survey_done_df[second_well])
pearson_corr(main_survey_done_df['1_trust2'], main_survey_done_df[first_well])
# pointbiserial_corr(main_survey_done_df['drive_name'].str[3], main_survey_done_df['3_total'])
# pearson_corr(main_survey_done_df['drive_name'].str[3], main_survey_done_df['3_total'])
# pearson_corr(main_survey_done_df['drive_name'].str[2], main_survey_done_df['2_Intention'])
pointbiserial_corr(main_survey_done_df[elem_intent], main_survey_done_df['2_Intention'])
pointbiserial_corr(main_survey_done_df['drive_name'].str[2], main_survey_done_df['2_Intention'])
pearson_corr(negative_deed_scooter_df['3_perception'], negative_deed_scooter_df[second_well])
# spearman_corr(main_survey_done_df['drive_name'].str[2], main_survey_done_df['2_Intention'])
# main_survey_done_df['action'] = main_survey_done_df['drive_name'].str[2]
# chi_square(main_survey_done_df, 'action', '2_Intention')


####################################################################
# bar plot for intention vs robot behavior
# a = gooddeed_df['2_Intention'].value_counts()['0']
# b = gooddeed_df['2_Intention'].value_counts()['1']
# c = baddeed_df['2_Intention'].value_counts()['0']
# d = baddeed_df['2_Intention'].value_counts()['1']
# bar_plot([a, c], [b, d], ['gooddeed', 'baddeed'], label1='neg', label2='pos', xlabel='x', ylable='y', title = 'title', name = 'name')

#######################################################################
# to check correlation between mental wellbeing and intention
# new_df = main_survey_done_df[~(main_survey_done_df['subID'].isin(['32','23']))]
# df_dup = pd.concat([mental_wellbeing_df, mental_wellbeing_df])
# # pearson_corr(new_df['2_Intention'], df_dup['total_well'])

#####################################################3
#correlation matrix for wellbeing
#####################################################
# df_elem, df_factor = make_corr_df(main_survey_done_df, mode='1')
#
# corr_elem_matrix = df_elem.corr()
# print(corr_elem_matrix)
# corr_plot(df_elem, corr_elem_matrix, title='all_element_')
# # plt.matshow(corr_elem_matrix)
# # plt.show()
# corr_factor_matrix = df_factor.corr()
# print(corr_factor_matrix)
# corr_plot(df_factor, corr_factor_matrix, title='factors_')
#
#

###########################################################################################
# a = pd.concat([main_survey_done_df['1_trust1'], main_survey_done_df['3_trust1']])
# b = pd.concat([main_survey_done_df['1_trust2'], main_survey_done_df['3_trust2']])
# new_df = pd.concat([a,b], axis=1)
# col = ['trust1', 'trust2']
# new_df.columns = col
# print(new_df)
#
# for i in new_df.columns:
#     new_df[i] = pd.to_numeric(new_df[i])
# plt.figure(figsize=(12, 8))
# sns.regplot(x=new_df['trust1'], y=new_df['trust2'], line_kws={"color": "r", "alpha": 0.7, "lw": 5})
# plt.savefig('usertrust_vs_trust' + ".png")
# plt.show()

###################################
#checking the data for close loop test the ones who has our policy condition vs ones who always yield or unyield and aligned ones
####################################
ourcodition_df = categorize_CloseLoop(main_survey_done_df)
alwaysYielding = positive_deed_scooter_df
alwaysUyielding = negative_deed_scooter_df
print(len(ourcodition_df))
t_test(ourcodition_df['1_total'], ourcodition_df['3_total'], mode='less')
t_test(alwaysYielding['1_total'], alwaysYielding['3_total'], mode='less')
t_test(alwaysUyielding['1_total'], alwaysUyielding['3_total'], mode='less')
t_test(align_df['1_total'], align_df['3_total'], mode='less')

# mean1 = ourcodition_df['1_total'].mean()
# mean2 = ourcodition_df['3_total'].mean()
# std1 = ourcodition_df['1_total'].std()
# std2 = ourcodition_df['3_total'].std()
# fig, ax = plt.subplots()
#
# # Create arrays for the x-values and error values
# x = np.array([1, 2])
# means = np.array([mean1, mean2])
# stds = np.array([std1, std2])
#
# # Plot the means as points with error bars
# ax.errorbar(x, means, yerr=stds, fmt='o')
#
# # Set the x-axis ticks and labels
# ax.set_xticks(x)
# ax.set_xticklabels(['Column 1', 'Column 2'])
#
# # Add a title and axis labels
# ax.set_title('Mean Values with Standard Deviations')
# ax.set_xlabel('Columns')
# ax.set_ylabel('Mean')
# print(mean1)
# print(mean2)
# # Display the plot
# plt.show()

###############
# demographics
###############
print(demographic_percentage(post_survey_df_nofilter, '4'))
print(demographic_percentage(post_survey_df_nofilter, '3'))
print(demographic_percentage(post_survey_df_nofilter, '5'))
print(demographic_percentage(post_survey_df_nofilter, '6'))





# bin_num = 6
# bins = [0.999 + (x / bin_num) * (bin_num + 1 + 0.001) for x in range(bin_num + 1)]
# print(bins)
# # define the labels for each category
# labels = list(range(0, bin_num))
# print(labels)