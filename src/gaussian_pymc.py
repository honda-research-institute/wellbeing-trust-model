from make_data_dbn import *
import numpy as np
import pymc3 as pm
# use np.int64 instead of np.int




# folder_path = './Mturk/data/'
folder_path = './Prolific/data_all/'
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)
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



df = make_dbn_data(main_survey_done_df)

data = df

df[("al", 0)] = df['C'] = (df[("a", 0)] == df[("I", 0)]).astype(int)

# t_0 =  df[("t", 0)].values.tolist()
# t_1 = df[("t", 1)].values.tolist()
# a_0 = df[("a", 0)].values.tolist()
# al_0 = df[("al", 0)].values.tolist()
# print(al_0)

# Define the model
with pm.Model() as model:

    mean_prior = pm.Uniform('mean_prior', lower=0, upper=300, shape=1)
    # sd_prior = pm.Uniform('sd_prior', lower=0, upper=300,)
    # mean_prior = pm.HalfNormal('mean_prior', sigma=1, shape=3)
    sd_prior = pm.HalfNormal('sd_prior', sigma=1)
    # Define the weights for the linear combination
    weights = pm.HalfNormal('weights', sigma=1, shape=3)

    # Define the mean as a linear combination of the parent nodes with the learned weights
    mean = pm.Deterministic('mean', weights[0] * df.iloc[:, 0] + weights[1] * df.iloc[:, 4] + weights[2] * df.iloc[:, 6] + mean_prior)

    # Define the standard deviation of the Gaussian distribution with a prior distribution
    sd = pm.Normal('sd', mu=sd_prior, sd=1)

    # Define the Gaussian distribution with the mean and standard deviation
    t_1 = pm.Normal('t_1', mu=mean, sd=sd, observed=df.iloc[:, 7])

    # Perform inference
    trace = pm.sample(4000, tune = 4000, chains=2, target_accept = .99)

#we then can use these variables in the dbn model or using them to tabulate cpds
    weights_mean = trace['weights'].mean(axis=0)
    mean_prior_mean = trace['mean_prior'].mean(axis=0)
    sd_prior_mean = trace['sd_prior'].mean()
    sd_mean = trace['sd'].mean()

print('weights_mean:', weights_mean)
print('mean_prior_mean:', mean_prior_mean)
print('sd_prior_mean:', sd_prior_mean)
print('sd_mean:', sd_mean)




