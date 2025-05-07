from make_data_dbn import *
import numpy as np
import pymc3 as pm
# use np.int64 instead of np.int
import arviz as az



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


#getting the prepared data for the model
df = make_sem_data_well_ps(main_survey_done_df)
df_all =  make_sem_data_well_ps_detail(main_survey_done_df)

data = df
#defining the model
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    alpha2 = pm.Normal('alpha2', mu=0, sd=10)
    beta3 = pm.Normal('beta3', mu=0, sd=10)
    beta4 = pm.Normal('beta4', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value
    mu_ps = pm.Deterministic('mu_ps', alpha + beta1 * data['action'])
    mu_w2 = pm.Deterministic('mu_w2', alpha2 + beta3 * data['wellbeing_t0'] + beta4 * data['perceived_sat_t1'])

    # Likelihoods
    ps = pm.Normal('ps', mu=mu_ps, sd=sigma, observed=data['perceived_sat_t1'])
    w2 = pm.Normal('w2', mu=mu_w2, sd=sigma, observed=data['wellbeing_t1'])

    # Inference
    trace = pm.sample(1000, tune=1000)


#
# # Generate posterior predictive samples for a range of values of ps
# ps_range = np.linspace(data['perceived_sat_t1'].min(), data['perceived_sat_t1'].max(), num=100)
# posterior_predictive_w2 = pm.sample_posterior_predictive(trace, var_names=['mu_w2'], samples=1000, keep_size=True, model=model, )
#
# # Plot the posterior predictive distribution for each value of ps
# az.plot_dist(posterior_predictive_w2['mu_w2'], point_estimate='mean', hdi_prob=0.95, color='C1', label='Posterior predictive distribution')
#
# # Add the observed data points
# plt.scatter(data['perceived_sat_t1'], data['wellbeing_t1'], alpha=0.5, label='Observed data')
#
# # Set the plot title and axis labels
# plt.title('Posterior predictive distribution of wellbeing_2 for different values of perceived_satisfaction')
# plt.xlabel('perceived_satisfaction')
# plt.ylabel('wellbeing_2')
#
# # Show the plot
# plt.show()


##### I tried to plot it however didn't debug it (this one doesn't work now)
# Generate posterior predictive samples
ppc = pm.sample_posterior_predictive(trace, model=model)

# Extract samples of perceived satisfaction and predicted wellbeing_2
ps_samples = ppc['ps']
w2_samples = ppc['w2']

# Calculate the mean predicted wellbeing_2 for each value of perceived satisfaction
ps_range = np.linspace(data['perceived_sat_t1'].min(), data['perceived_sat_t1'].max(), num=100)
mean_w2 = []
for ps_val in ps_range:
    # Select the indices of the posterior samples where perceived satisfaction is equal to ps_val
    idx = np.where(ps_samples == ps_val)[0]
    # Calculate the mean predicted wellbeing_2 for these samples
    mean_w2.append(np.mean(w2_samples[idx]))

# Plot the relationship between perceived satisfaction and predicted wellbeing_2
plt.plot(ps_range, mean_w2)
plt.xlabel('Perceived Satisfaction')
plt.ylabel('Predicted Wellbeing_2')
plt.show()



