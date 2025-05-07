from make_data_dbn import *
import numpy as np
from scipy.stats import norm
import pandas as pd
import semopy




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
mean_wellbeing_robot(main_survey_df)
mean_wellbeing_scooter(main_survey_df)
mean_wellbeing_robot(main_survey_done_df)
mean_wellbeing_scooter(main_survey_done_df)
gooddeed_df, baddeed_df, align_df, notalign_df, positive_deed_scooter_df, negative_deed_scooter_df, pos_intention_df, neg_intention_df = make_categories_main(main_survey_df)



df = make_sem_data_well_ps(main_survey_done_df)
df_all =  make_sem_data_well_ps_detail(main_survey_done_df)
# print(df)



# Load data
# data = df
data = df_all

#defining the model
model_spec= '''

    # Measurement model 
    # perceived_sat_t0 =~ perceived_sat_t0
    # perceived_sat_t1 =~ perceived_sat_t1
    # wellbeing_t0 =~ wellbeing_t0
    # wellbeing_t1 =~ wellbeing_t1
    wellbeing_2 =~ p1_2 + p2_2 + s1_2 + s2_2 + s3_2 + w_2 + t_2
    wellbeing_1 =~ p1_1 + p2_1 + s1_1 + s2_1 + s3_1 + w_1 + t_1
    # Structural model
    
    wellbeing_2 ~ perceived_satisfaction + alignment + wellbeing_1
    perceived_satisfaction ~ action 
    
    
    # wellbeing_2 ~ alignment + wellbeing_1
    # perceived_satisfaction ~ action + wellbeing_2 
    # 
    # wellbeing ~ alignment 
    # perceived_sat_t1 ~ wellbeing + action
    # 
    # wellbeing_t1 ~ perceived_sat_t1 + wellbeing_t0 + alignment + trust
    # perceived_sat_t1 ~ perceived_sat_t0 + action
    
    # wellbeing_t1 ~  wellbeing_t0 + alignment + trust
    # perceived_sat_t1 ~ perceived_sat_t0 + wellbeing_t1 + action
    
    # wellbeing_t1 ~ perceived_sat_t1 
    # perceived_sat_t1 ~ action
    
    # wellbeing_t1 ~ alignment
    # perceived_sat_t1 ~ wellbeing_t1
    
    # wellbeing_t1 ~ perceived_sat_t1 + wellbeing_t0 + alignment
    # perceived_sat_t1 ~ action
    
    # wellbeing_t1 ~  wellbeing_t0 + alignment 
    # perceived_sat_t1 ~  wellbeing_t1 + action

    # wellbeing_t0 =~ wellbeing_t0
    # perceived_sat_t0 =~ perceived_sat_t0
    # action =~ action
    # alignment =~ alignment
    
    # perceived_sat_t1 ~ perceived_sat_t0 + action
    # wellbeing_t1 ~ perceived_sat_t1 + wellbeing_t0 + alignment

    

'''

# Create semopy model
model = semopy.Model(model_spec)

# Fit model
res = model.fit(data)
print(res)

g = semopy.semplot(model, "pd.png")
plt.show()

stats = semopy.calc_stats(model)
print(stats.T)
# Print standardized coefficients
print(model.inspect())

