from make_data_dbn import *
import numpy as np
from scipy.stats import norm
import pandas as pd
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.metrics import log_likelihood_score
from pgmpy.metrics.bn_inference import BayesianModelProbability
# from pgmpy.inference import VariableElimination
from pgmpy.inference import DBNInference
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import GroupKFold, KFold
from pgmpy.sampling import BayesianModelSampling
from stat_tests import *




folder_path = './Prolific/data_all/'
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 1000)
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
df_with_turkid = make_df_participant_based(main_survey_done_df)
gooddeed_df, baddeed_df, align_df, notalign_df, positive_deed_scooter_df, negative_deed_scooter_df, pos_intention_df, neg_intention_df = make_categories_main(main_survey_df)
swap_df = pd.read_csv('swap_action_3seq.csv')
# print(swap_df)
def is_align(x1, x2):
    if x1 == x2:
        return .999
    else:
        return 0.001


df_seq = make_dbn_data_unique_turkID(df_with_turkid, swap_df)

df_seq['al_1'] = (df_seq['a_1'] == df_seq['I_1']).astype(int)

#defining the two network bn1T is SC_dbn and bn2T is RC_dbn

bn1T = BN([('t_0', 't_1'), ('w_0', 'w_1'), ('ow_0', 'ow_1'), ('w_0', 'I_1'), ('a_1', 'ow_1'), ('a_1', 't_1'), ('I_1', 'al_1'), ('a_1', 'al_1'), ('al_1','w_1'), ('al_1',
                                                                                                                                                                't_1'),
           ('t_0', 'w_0'), ('t_1', 'w_1')])

bn2T = BN([('t_1', 't_2'), ('w_1', 'w_2'), ('ow_1', 'ow_2'), ('oa_2', 'w_2'), ('t_1', 'w_1'), ('t_2', 'w_2')])

#adding the cpds for alignment

values = [[1-is_align(0, 0), 1-is_align(0, 1), 1-is_align(1, 0), 1-is_align(1, 1)],
          [is_align(0, 0), is_align(0, 1), is_align(1, 0), is_align(1, 1)]]
tabular_cpd_al_1 = TabularCPD(variable='al_1', variable_card=2, values=values, evidence=['a_1', 'I_1'], evidence_card=[2, 2])


bn1T.add_cpds(tabular_cpd_al_1)


#add cpds of t_1
mode = 'none'
# mode = "gaussian"
# dbn.add_cpds(tabular_cpd_t_1)
# bn.add_cpds(tabular_cpd_t_1_bn)



data_bn = df_seq
# print(data_bn)
# data_bn['mTurkID'] = main_survey_done_df['mTurkID']
# print(df_bn)

# define the number of folds for cross-validation
k = 5
#leave one out
# k = 299

# split the data into k folds
kf = KFold(n_splits=k)
idxs = data_bn.index
# iterate over the folds and train/test the model
likelihoodscores1T = []
likelihoodscores2T = []
folds = list(kf.split(data_bn))
mse_list = []
rmse_list = []
corr_list = []
accuracy_list = []
n_samples = 1
col_to_include = ['w_1', 't_1', 'I_1', 'ow_1']
# accuracy_dict = {col: [] for col in col_to_include}
n_iterations = 1  # Number of times to repeat the accuracy computation
overall_accuracy_dict = {col: [] for col in col_to_include}  # To store accuracy for each column across iterations

for _ in range(n_iterations):
    accuracy_dict = {col: [] for col in col_to_include}






    for train_index, test_index in folds:
        
        # split the data into train and test sets
        train_data = data_bn.iloc[train_index]
        test_data = data_bn.iloc[test_index]

        # add cpds of t_1
        mode = 'none'
        # mode = "gaussian"
        # bn.add_cpds(tabular_cpd_t_1_bn)

        # fit the model to the train data
        # bn.fit(train_data.iloc[:, 0:9], estimator=BayesianEstimator, prior_type="dirichlet")
        for i in train_data.columns:
            train_data[i] = pd.to_numeric(train_data[i])

        for i in test_data.columns:
            test_data[i] = pd.to_numeric(test_data[i])

        bn1T.fit(train_data.iloc[:, 0:9], estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=20)
        bn2T.fit(train_data.iloc[:, 6:], estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=20)

        # bn1T.fit(train_data.iloc[:, 0:9])
        # bn2T.fit(train_data.iloc[:, 6:])
        #using forward sample to get inference of data (SC_DBN) of a similar size of test data
        inference = BayesianModelSampling(bn1T)
        # print('size', test_data.shape[0])
        sm = inference.forward_sample(size=test_data.shape[0])


        all_samples = []
        for _ in range(n_samples):
            sm1 = inference.forward_sample(size=test_data.shape[0])
            sm1 = sm1.set_index(test_data.index)
            all_samples.append(sm1)
            
        # Convert list of samples to a DataFrame and compute the mean
        all_samples_df = pd.concat(all_samples).groupby(level=0).mean()
        all_samples_df = all_samples_df.loc[test_data.index]

        # Sort both DataFrames by the relevant columns
        # sorted_test_data = test_data[col_to_include].sort_values(by=col_to_include)
        # sorted_samples_df = all_samples_df[col_to_include].sort_values(by=col_to_include)
        sorted_test_data = test_data[col_to_include].apply(np.sort, axis=0)
        sorted_samples_df = all_samples_df[col_to_include].apply(np.sort, axis=0)
        # Reset the indices to ensure they are aligned
        sorted_test_data = sorted_test_data.reset_index(drop=True)
        sorted_samples_df = sorted_samples_df.reset_index(drop=True)
        # print(sorted_samples_df[col_to_include].round(), 'test', sorted_test_data[col_to_include].round())

        # bn.fit(train_data.iloc[:, 0:9])
        # bn.fit(train_data.iloc[:, 0:9])
        # a = .00001
        # # a = 0
        # for node in bn2T.nodes():
        #     cpd = bn2T.get_cpds(node)
        #     values = np.round(cpd.values, 6) + a
        #     cpd.values = values
        #     cpd.normalize()
        #

        # compute the log-likelihood of the model on the test data
    
            # print(test_data[i].dtype)
        # print(train_data.iloc[:, 0:9])
        # print(test_data.iloc[:, 0:9])
        # print(train_data['w_1'].dtype)
        log_likelihood1T = log_likelihood_score(bn1T, test_data.iloc[:, 0:9])
        log_likelihood2T = log_likelihood_score(bn2T, test_data.iloc[:, 6:])
        # print(log_likelihood)
    #Using the infered sample data to compute rmse, mse and correlation
        
        # append the log-likelihood score to the list of scores
        likelihoodscores1T.append(log_likelihood1T)
        # print(likelihoodscores1T)
        likelihoodscores2T.append(log_likelihood2T)
        sm = sm.set_index(test_data.index)
        # print('test index', test_data.index)
        # print(test_data[col_to_include], sm[col_to_include])
        # print(test_data[col_to_include].equals(sm[col_to_include]))

        
        # Compute accuracy by comparing test_data and sm
        # correct_predictions = (test_data[col_to_include].round() == sm[col_to_include].round()).sum().sum()
        # total_predictions = np.product(test_data[col_to_include].shape)
        # accuracy = correct_predictions / total_predictions
        # accuracy_list.append(accuracy)

        #altenative
        print(sorted_samples_df[col_to_include], sorted_test_data[col_to_include])
        for col in col_to_include:

            correct_predictions = (sorted_test_data[col].round() == sorted_samples_df[col].round()).sum().sum()
            
            # total_predictions = np.product(test_data[col].shape)
            total_predictions = len(sorted_test_data[col])
            print('correct_predict', correct_predictions, total_predictions)
            accuracy = correct_predictions / total_predictions
            
            # accuracy_list.append(accuracy)
            accuracy_dict[col].append(accuracy)
            # print('accuracy list', accuracy_list)
    
    
        #mse rmse cor with more sample 
        mse_fold = np.mean((((sorted_test_data[col_to_include].round()-1)/6) - ((sorted_samples_df[col_to_include].round())/6))**2)
        # mse_fold = np.mean((((test_data[col_to_include]-1)/6) - ((sm[col_to_include]-1)/6))**2)
        # print(test_data[col_to_include]-sm[col_to_include])
        # print(sm[col_to_include])
        # print(mse_fold)
        rmse_fold = np.sqrt(mse_fold)
        cor_fold = sorted_test_data[col_to_include].round().corrwith(sorted_samples_df[col_to_include].round(), axis=0)
        mse_list.append(mse_fold)
        # for i in col_to_include:
        #     print(t_test(test_data[i], sm[i]))
        rmse_list.append(rmse_fold)
        corr_list.append(cor_fold)
        
        #mse rmse cor
        # mse_fold = np.mean((((test_data[col_to_include].sort_values(col_to_include)-1)/6) - ((sm1[col_to_include].sort_values(col_to_include)-1)/6))**2)
        # # mse_fold = np.mean((((test_data[col_to_include]-1)/6) - ((sm[col_to_include]-1)/6))**2)
        # # print(test_data[col_to_include]-sm[col_to_include])
        # # print(sm[col_to_include])
        # # print(mse_fold)
        # rmse_fold = np.sqrt(mse_fold)
        # cor_fold = test_data[col_to_include].corrwith(sm1[col_to_include], axis=0)
        # mse_list.append(mse_fold)
        # for i in col_to_include:
        #     print(t_test(test_data[i], sm[i]))
        # rmse_list.append(rmse_fold)
        # corr_list.append(cor_fold)



    # compute the average log-likelihood score
    sum_score1T = sum(likelihoodscores1T)
    print(f"sum log-likelihood score 1T: {sum_score1T}")

    sum_score2T = sum(likelihoodscores2T)
    print(f"sum log-likelihood score 2T: {sum_score2T}")
    print(sum_score1T, sum_score2T)
    ave_mse = np.mean(mse_list, axis=0)
    # print(corr_list)
    ave_rmse = np.mean(rmse_list, axis=0)
    ave_corr = np.mean(corr_list, axis=0)
    # average_accuracy = np.mean(accuracy_list)
    # print(ave_mse, ave_rmse, ave_corr)
    # print(f"Average MSE: {ave_mse}, Average RMSE: {ave_rmse}, Average Correlation: {ave_corr}, Average Accuracy: {average_accuracy}")
    print(f"Average MSE: {ave_mse}, Average RMSE: {ave_rmse}, Average Correlation: {ave_corr}")

    for col in col_to_include:
        average_accuracy = np.mean(accuracy_dict[col])
        # print(f"Average Accuracy for {col}: {average_accuracy}")
        overall_accuracy_dict[col].append(average_accuracy)

# Compute and print the final average accuracy across all iterations
for col in col_to_include:
    final_average_accuracy = np.mean(overall_accuracy_dict[col])
    print(f"Final Average Accuracy for {col} across {n_iterations} iterations: {final_average_accuracy}")

a = 0
with open("network_edges_bn1T_kfold"+mode+".txt", "w") as text_file:
    for node in bn1T.nodes():
        cpd = bn1T.get_cpds(node)
        try:
            values = np.round(cpd.values, 6) + a
            cpd.values = values
            cpd.normalize()
            print(f"CPD of {node}: \n{cpd.to_df()}", file=text_file)
        except AttributeError:
            print(f"CPD of {node}: \n{cpd}", file=text_file)







