from make_data_dbn import *
import numpy as np
from scipy.stats import norm
import networkx as nx

from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.metrics import log_likelihood_score
from pgmpy.estimators import BayesianEstimator
import pycid


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


def is_align(x1, x2):
    if x1==x2:
        return .999
    else:
        return 0.001


df_seq = make_dbn_data_unique_turkID(df_with_turkid, swap_df)

df_seq['al_1'] = (df_seq['a_1'] == df_seq['I_1']).astype(int)

col_to_normalize = ['t_0', 'w_0', 'ow_0', 't_1', 'w_1', 'ow_1', 't_2', 'w_2', 'ow_2']
# col_to_normalize = ['U_1', 'U_2']
# print(df_seq[col_to_normalize].dtypes)
df_seq[col_to_normalize] = df_seq[col_to_normalize].astype(float).apply(lambda x: (x) / 5)
# print(df_seq)

#Defining the utilities, U2 is the utility for bn2T (RC-dbn) and U1 is for bn1T (SC_DBN)
# df_seq.insert(6, 'U_1', df_seq['t_1'].astype(float))
df_seq['U_2'] = df_seq['t_2'].astype(float)
# df_bn['U_2'] = df_bn['ow_1'].astype(float)

# print((df_bn['a_0'] == 1).sum(), (df_bn['oa_0'] == 1).sum())
# print(df_bn['a_0'], df_bn['oa_0'], df_bn['ow_1'])
# print(df_bn['w_1'], df_bn['ow_1'])

# def my_function(row):
#     if row['a_0'] == 1:
#         return row['ow_1'] + row['w_1'] - 0.0000000000000001
#     else:
#         return row['ow_1'] + row['w_1']

############################when U1 = t_1
# def my_function(row):
#     if row['a_0'] == 1:
#         return row['t_1']
#     else:
#         return row['t_1']


################### when U1 = w_1
def my_function(row):
    if row['a_1'] == 1:
        return row['w_1']
    else:
        return row['w_1']

############# when U1 = w_1 +ow_1 + c
# def my_function(row):
#     if row['a_0'] == 1:
#         return row['ow_1'] + row['w_1'] - 0.0000000000000002
#     else:
#         return row['ow_1'] + row['w_1']

# we can define various c
# c = 1.1/5
# c = 1.000000000000001/5
# c = 1.000000000000001/5
# # c = .99999999999999939/5
# def my_function(row):
#     if row['a_1'] == 1:
#         return row['ow_1'] + row['w_1'] - c
#     else:
#         return row['ow_1'] + row['w_1']
#
#
# # df_seq['U_1'] = df_seq.apply(my_function, axis=1)

df_seq.insert(6, 'U_1', df_seq.apply(my_function, axis=1))


#normalized
# c= 5/6
# def my_function(row):
#     if row['a_1'] == 1:
#         return ((row['ow_1']-1) + (row['w_1']-1))/6 - c
#     else:
#         return ((row['ow_1']-1) + (row['w_1']-1))/6
#
#
# # df_seq['U_1'] = df_seq.apply(my_function, axis=1)
# df_seq.insert(6, 'U_1', df_seq.apply(my_function, axis=1))

# print(df_seq)
# defining the dbn networks with utility node and information link if we want to have evidence
bn1T = BN([('t_0', 't_1'), ('w_0', 'w_1'), ('ow_0', 'ow_1'), ('w_0', 'I_1'), ('a_1', 'ow_1'), ('a_1', 't_1'), ('I_1', 'al_1'), ('a_1', 'al_1'), ('al_1','w_1'), ('al_1',
                                                                                                                                                                't_1'),
           ('t_0', 'w_0'), ('t_1', 'w_1'), ('w_1', 'U_1'), ('w_0', 'a_1')])

bn2T = BN([('t_1', 't_2'), ('w_1', 'w_2'), ('ow_1', 'ow_2'), ('oa_2', 'w_2'), ('t_1', 'w_1'), ('t_2', 'w_2'), ('t_2', 'U_2')])


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

# a = .00001
a = 0
# bn.fit(data_bn)

#estimating the cpds
bn1T.fit(data_bn.iloc[:, 0:10], estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=20)
bn2T.fit(data_bn.iloc[:, 6:], estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=20)
# for node in bn1T.nodes():
#     cpd = bn1T.get_cpds(node)
#     values = np.round(cpd.values, 6) + a
#     cpd.values = values
#     cpd.normalize()


####### defining the cid network, it is similar to dbn but we need to define decision node and utility node as well. The current version consider w_0 as evidence for
# decision making, we can remove it ('w_0', 'a_1') in the lack of evidence or make another evidence instead of this. Also, this current version is for the version that
# utility is equal to w_1 if it is something else we need to replace ('w_1', 'U_1') with the associate nodes for utility

# ###CID  cid1T is related to SC_dbn and cid2T is related to RC_dbn
cid1T = pycid.CID([('t_0', 't_1'), ('w_0', 'w_1'), ('ow_0', 'ow_1'), ('w_0', 'I_1'), ('a_1', 'ow_1'), ('a_1', 't_1'), ('I_1', 'al_1'), ('a_1', 'al_1'), ('al_1','w_1'), ('al_1',
                                                                                                                                                                't_1'),
           ('t_0', 'w_0'), ('t_1', 'w_1'), ('w_1', 'U_1'), ('w_0', 'a_1')],
                decisions=["a_1"],
                utilities=["U_1"],
                )

cid2T = pycid.CID([('t_1', 't_2'), ('w_1', 'w_2'), ('ow_1', 'ow_2'), ('oa_2', 'w_2'), ('t_1', 'w_1'), ('t_2', 'w_2'), ('t_2', 'U_2')],
                decisions=["oa_2"],
                utilities=["U_2"],
                )

##adding the estimated cpds from the dbn to cid
for node in bn1T.nodes():
    cpd1 = bn1T.get_cpds(node)
    tab_cpd = TabularCPD(variable=node, variable_card=cpd1.variable_card, values=cpd1.values.flatten().reshape((cpd1.variable_card, np.prod(cpd1.cardinality[1:]))),
                         evidence=cpd1.get_evidence()[::-1], evidence_card=cpd1.cardinality[1:])
    # print(tab_cpd)
    cid1T.add_cpds(tab_cpd)


for node in bn2T.nodes():
    cpd2 = bn2T.get_cpds(node)
    tab_cpd = TabularCPD(variable=node, variable_card=cpd2.variable_card, values=cpd2.values.flatten().reshape((cpd2.variable_card, np.prod(cpd2.cardinality[1:]))),
                         evidence=cpd2.get_evidence()[::-1], evidence_card=cpd2.cardinality[1:])
    # print(tab_cpd)
    cid2T.add_cpds(tab_cpd)


# cid1T.draw()
# cid2T.draw()

#####finding optimal policy of cid
# solution = cid1T.solve()
solution1 = cid1T.optimal_policies()
print(solution1)

solution2 = cid2T.optimal_policies()
print(solution2)


# cid.model.update(solution)
# cid1T.impute_optimal_policy()
# ex = cid1T.expected_utility({})
# print(ex)
# #
# cid2T.impute_optimal_policy()
# ex2 = cid2T.expected_utility({})
# print(ex2)



# voi
# pycid.admits_voi_list(cid1T, 'a_1')
# cid1T.draw_property(lambda node:
#                         pycid.admits_voi(cid1T, 'a_1', node))
#
# pycid.admits_voi_list(cid2T, 'oa_2')
# cid2T.draw_property(lambda node:
#                         pycid.admits_voi(cid2T, 'oa_2', node))
#


#ri
# pycid.admits_ri_list(cid, 'a_0')
# cid.draw_property(lambda node:
#                         pycid.admits_ri(cid, 'a_0', node))

#voc
# pycid.admits_voc_list(cid)
# cid.draw_property(lambda node:
#