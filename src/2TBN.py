from make_data_dbn import *
import numpy as np
from scipy.stats import norm
import networkx as nx

from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.metrics import log_likelihood_score
from pgmpy.estimators import BayesianEstimator
# import pycid
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from sklearn.model_selection import KFold
import networkx as nx
from pgmpy.estimators import ExhaustiveSearch
from pgmpy.inference import BeliefPropagation, CausalInference
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State


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




#defining the two network bn1T is SC_dbn and bn2T is RC_dbn

bn1T = BN([('t_0', 't_1'), ('w_0', 'w_1'), ('ow_0', 'ow_1'), ('w_0', 'I_1'), ('a_1', 'ow_1'), ('a_1', 't_1'), ('I_1', 'al_1'), ('a_1', 'al_1'), ('al_1','w_1'), ('al_1',
                                                                                                                                                                't_1'),
           ('t_0', 'w_0'), ('t_1', 'w_1')])

bn2T = BN([('t_1', 't_2'), ('w_1', 'w_2'), ('ow_1', 'ow_2'), ('oa_2', 'w_2'), ('t_1', 'w_1'), ('t_2', 'w_2')])


values = [[1-is_align(0, 0), 1-is_align(0, 1), 1-is_align(1, 0), 1-is_align(1, 1)],
          [is_align(0, 0), is_align(0, 1), is_align(1, 0), is_align(1, 1)]]
tabular_cpd_al_1 = TabularCPD(variable='al_1', variable_card=2, values=values, evidence=['a_1', 'I_1'], evidence_card=[2, 2])


bn1T.add_cpds(tabular_cpd_al_1)


#add cpds of t_1
mode = ''
# mode = "gaussian"
# dbn.add_cpds(tabular_cpd_t_1)
# bn.add_cpds(tabular_cpd_t_1_bn)



data_bn = df_seq
# print(data_bn)
# a = .00001
# bn.fit(data_bn)
#fitting the data to estimate cpds using bayesian estimator
bn1T.fit(data_bn.iloc[:, 0:9], estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=20)
bn2T.fit(data_bn.iloc[:, 6:], estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=20)


#Saving learned tabular cpds in the text files
with open("CPDs_bn1T"+mode+".txt", "w") as text_file:
    for node in bn1T.nodes():
        cpd = bn1T.get_cpds(node)
        try:
            values = np.round(cpd.values, 4)
            cpd.values = values
            print(f"CPD of {node}: \n{cpd.to_df()}", file=text_file)
        except AttributeError:
            print(f"CPD of {node}: \n{cpd}", file=text_file)

with open("CPDs_bn2T"+mode+".txt", "w") as text_file:
    for node in bn2T.nodes():
        cpd = bn2T.get_cpds(node)
        try:
            values = np.round(cpd.values, 4)
            cpd.values = values
            print(f"CPD of {node}: \n{cpd.to_df()}", file=text_file)
        except AttributeError:
            print(f"CPD of {node}: \n{cpd}", file=text_file)



# inference = VariableElimination(bn1T)
# phi_query = inference.max_marginal(variables=['w_1'], evidence={'al_0': 1})
# phi_query = inference.query(variables=['w_0'])
# #
# print(phi_query)

# inference = BeliefPropagation(bn1T)
# phi_query = inference.query(variables=['t_1'], evidence={'al_1': 1, 'a_1':1})
# #
# print(phi_query)

#############################################  bn1T
#inferring trust, wellbeing and other's wellbeing over 10 time step with different evidence (fixing them)
belief_propagation = BeliefPropagation(bn1T)
t_1_exp = [sum([x * y for x, y in zip([0, 1/5, 2/5, 3/5, 4/5, 1], belief_propagation.query(variables=['t_1']).values.tolist())])]
w_1_exp = [sum([x * y for x, y in zip([0, 1/5, 2/5, 3/5, 4/5, 1], belief_propagation.query(variables=['w_1']).values.tolist())])]
ow_1_exp = [sum([x * y for x, y in zip([0, 1/5, 2/5, 3/5, 4/5, 1], belief_propagation.query(variables=['ow_1']).values.tolist())])]
t_1_mat = np.zeros(6)
w_1_mat = np.zeros(6)
ow_1_mat = np.zeros(6)
for i in range(10):
    belief_propagation = BeliefPropagation(bn1T)

    # t_1_exp.append(belief_propagation.query(variables=['t_1']))
    # w_1_exp.append(belief_propagation.query(variables=['w_1']))
    # ow_1_exp.append(belief_propagation.query(variables=['ow_1']))

    bpq_w = belief_propagation.query(variables=['w_1'], evidence={'al_1': 0})
    bpq_t = belief_propagation.query(variables=['t_1'],
                         evidence={'al_1': 0, 'a_1': 1})
    bpq_ow = belief_propagation.query(variables=['ow_1'], evidence={'a_1': 1})
    # print(bpq_t)
    V1 = [0, 1/5, 2/5, 3/5, 4/5, 1]
    V2 = bpq_t.values.tolist()
    Vw = bpq_w.values.tolist()
    Vow = bpq_ow.values.tolist()
    exp_t = sum([x * y for x, y in zip(V1, V2)])
    exp_w = sum([x * y for x, y in zip(V1, Vw)])
    exp_ow = sum([x * y for x, y in zip(V1, Vow)])

    t_1_exp.append(exp_t)
    w_1_exp.append(exp_w)
    ow_1_exp.append(exp_ow)
    t_1_mat = np.vstack([t_1_mat, V2])
    w_1_mat = np.vstack([w_1_mat, Vw])
    ow_1_mat = np.vstack([ow_1_mat, Vow])
    tab_cpd_t_0_up = TabularCPD(variable='t_0', variable_card=6, values=np.transpose([bpq_t.values.tolist()]))
    tab_cpd_ow_0_up = TabularCPD(variable='ow_0', variable_card=6, values=np.transpose([bpq_ow.values.tolist()]))
    bn1T.add_cpds(tab_cpd_t_0_up, tab_cpd_ow_0_up)
    # print(bn1T.get_cpds('t_0'))
    # bp = BeliefPropagation(bn1T)
    # bp.calibrate()
    # # print(bp.query(variables=['t_0']))
    # bpq_w0 = bp.query(variables=['w_1'])
    # # print(bn1T.get_cpds('w_0'))
    # # Vw2 = bpq_w0.values.tolist()
    # # exp_w = sum([x * y for x, y in zip(V1, Vw2)])
    # # bpq_w0 = belief_propagation.query(variables=['w_0'])
    # print(bpq_w0)

#######################################bn2T
# belief_propagation = VariableElimination(bn2T)
# t_1_exp = []
# w_1_exp = [sum([x * y for x, y in zip([0, 1/5, 2/5, 3/5, 4/5, 1], belief_propagation.query(variables=['w_2']).values.tolist())])]
# ow_1_exp = []
# t_1_mat = np.zeros(6)
# w_1_mat = np.zeros(6)
# ow_1_mat = np.zeros(6)
# for i in range(10):
#     belief_propagation = VariableElimination(bn2T)
#
#     bpq_w = belief_propagation.query(variables=['w_2'], evidence={'oa_2': 0})
#     bpq_t = belief_propagation.query(variables=['t_2'])
#     bpq_ow = belief_propagation.query(variables=['ow_2'])
#     # print(bpq_t)
#     V1 = [0, 1/5, 2/5, 3/5, 4/5, 1]
#     V2 = bpq_t.values.tolist()
#     Vw = bpq_w.values.tolist()
#     Vow = bpq_ow.values.tolist()
#     exp_t = sum([x * y for x, y in zip(V1, V2)])
#     exp_w = sum([x * y for x, y in zip(V1, Vw)])
#     exp_ow = sum([x * y for x, y in zip(V1, Vow)])
#
#     t_1_exp.append(exp_t)
#     w_1_exp.append(exp_w)
#     ow_1_exp.append(exp_ow)
#     t_1_mat = np.vstack([t_1_mat, V2])
#     w_1_mat = np.vstack([w_1_mat, Vw])
#     ow_1_mat = np.vstack([ow_1_mat, Vow])
#     tab_cpd_t_1_up = TabularCPD(variable='t_1', variable_card=6, values=np.transpose([bpq_t.values.tolist()]))
#     tab_cpd_ow_1_up = TabularCPD(variable='ow_1', variable_card=6, values=np.transpose([bpq_ow.values.tolist()]))
#     bn2T.add_cpds(tab_cpd_t_1_up, tab_cpd_ow_1_up)

##########################################alternative bn1T bn2T

# belief_propagation = BeliefPropagation(bn1T)
# t_1_exp = [sum([x * y for x, y in zip([0, 1/5, 2/5, 3/5, 4/5, 1], belief_propagation.query(variables=['t_1']).values.tolist())])]
# w_1_exp = [sum([x * y for x, y in zip([0, 1/5, 2/5, 3/5, 4/5, 1], belief_propagation.query(variables=['w_1']).values.tolist())])]
# ow_1_exp = [sum([x * y for x, y in zip([0, 1/5, 2/5, 3/5, 4/5, 1], belief_propagation.query(variables=['ow_1']).values.tolist())])]
# t_1_mat = np.zeros(6)
# w_1_mat = np.zeros(6)
# ow_1_mat = np.zeros(6)
# for i in range(5):
#     belief_propagation = BeliefPropagation(bn1T)
#
#     bpq_w = belief_propagation.query(variables=['w_1'], evidence={'al_1': 0})
#     bpq_t = belief_propagation.query(variables=['t_1'],
#                          evidence={'al_1': 0, 'a_1': 0})
#     bpq_ow = belief_propagation.query(variables=['ow_1'], evidence={'a_1': 0})
#     # print(bpq_t)
#     V1 = [0, 1/5, 2/5, 3/5, 4/5, 1]
#     V2 = bpq_t.values.tolist()
#     Vw = bpq_w.values.tolist()
#     Vow = bpq_ow.values.tolist()
#     exp_t = sum([x * y for x, y in zip(V1, V2)])
#     exp_w = sum([x * y for x, y in zip(V1, Vw)])
#     exp_ow = sum([x * y for x, y in zip(V1, Vow)])
#
#     t_1_exp.append(exp_t)
#     w_1_exp.append(exp_w)
#     ow_1_exp.append(exp_ow)
#     t_1_mat = np.vstack([t_1_mat, V2])
#     w_1_mat = np.vstack([w_1_mat, Vw])
#     ow_1_mat = np.vstack([ow_1_mat, Vow])
#     tab_cpd_t_0_up = TabularCPD(variable='t_1', variable_card=6, values=np.transpose([bpq_t.values.tolist()]))
#     tab_cpd_ow_0_up = TabularCPD(variable='ow_1', variable_card=6, values=np.transpose([bpq_ow.values.tolist()]))
#     bn2T.add_cpds(tab_cpd_t_0_up, tab_cpd_ow_0_up)
#
#     belief_propagation = VariableElimination(bn2T)
#
#     bpq_w = belief_propagation.query(variables=['w_2'], evidence={'oa_2': 0})
#     bpq_t = belief_propagation.query(variables=['t_2'])
#     bpq_ow = belief_propagation.query(variables=['ow_2'])
#
#     V2 = bpq_t.values.tolist()
#     Vw = bpq_w.values.tolist()
#     Vow = bpq_ow.values.tolist()
#     exp_t = sum([x * y for x, y in zip(V1, V2)])
#     exp_w = sum([x * y for x, y in zip(V1, Vw)])
#     exp_ow = sum([x * y for x, y in zip(V1, Vow)])
#
#     t_1_exp.append(exp_t)
#     w_1_exp.append(exp_w)
#     ow_1_exp.append(exp_ow)
#     t_1_mat = np.vstack([t_1_mat, V2])
#     w_1_mat = np.vstack([w_1_mat, Vw])
#     ow_1_mat = np.vstack([ow_1_mat, Vow])
#     tab_cpd_t_1_up = TabularCPD(variable='t_0', variable_card=6, values=np.transpose([bpq_t.values.tolist()]))
#     tab_cpd_ow_1_up = TabularCPD(variable='ow_0', variable_card=6, values=np.transpose([bpq_ow.values.tolist()]))
#     bn1T.add_cpds(tab_cpd_t_1_up, tab_cpd_ow_1_up)
#
print(t_1_exp, '\n', w_1_exp, '\n', ow_1_exp)
# inference = BayesianModelSampling(bn)
# # evidence = [State('al', 1)]
# print(inference.likelihood_weighted_sample(size=10))

# inference = BayesianModelSampling(bn)
# print(inference.forward_sample(size=10))
#both0
# [0.4929982643218722, 0.4169824856251244, 0.37758852399322707, 0.35701606234448047, 0.3462065159692914, 0.3404847372232679, 0.337427165483, 0.3357744967825989, 0.3348698782684856, 0.33436836589173513]
# [0.5352787511256242, 0.49619519873648543, 0.47629915945705514, 0.4660117645760866, 0.46064645826920503, 0.45782638272404164, 0.456330233179107, 0.455527371600573, 0.4550909059425954, 0.4548503782808191]
# [0.49379221663622413, 0.4485511214556563, 0.43026341379226385, 0.4230412831002494, 0.42025129889837815, 0.4191968420411494, 0.4188096685139356, 0.4186736142235569, 0.41862928723093684, 0.4186169398070904]
#both1
# [0.7143976046191889, 0.7649793442006916, 0.7975998948248891, 0.817541143611032, 0.8292071655454119, 0.8357370718896424, 0.8392048530197822, 0.8409165472082638, 0.8416643348230926, 0.8419121961473165]
# [0.6501252343356236, 0.6792230939769976, 0.6992995346693286, 0.7122481635954552, 0.7201994949340685, 0.7248857216022262, 0.7275360425864571, 0.7289642823633946, 0.7296853830489926, 0.730013694066431]
# [0.638719663274635, 0.6665663109825825, 0.682733494221473, 0.6921895443842756, 0.6977550421377802, 0.7010423316073298, 0.7029880533195597, 0.7041412330902531, 0.7048252876439977, 0.705231298928102]

#bn2T
#oa_21
# [0.6476946447221685, 0.6640454225056212, 0.6723533051960155, 0.6767143581287967, 0.6791275661367611, 0.6805435093742505, 0.6814216819254697, 0.6819923188761863, 0.6823765804767673, 0.6826419912313295]
# [0.6439511053252467, 0.6523927980528503, 0.6570882387295471, 0.6595904779711111, 0.6609663418111574, 0.6617635321313926, 0.6622515311180115, 0.6625651190292124, 0.6627745009439785, 0.6629182553535093]
# [0.5943596950574914, 0.6097531889168286, 0.6179180528991234, 0.6222635694309935, 0.6245904327755052, 0.6258436427598952, 0.6265218428403796, 0.6268902645954744, 0.6270910145045151, 0.6272006738111018]
#oa_20
# [0.6476946447221685, 0.6640454225056212, 0.6723533051960155, 0.6767143581287967, 0.6791275661367611, 0.6805435093742505, 0.6814216819254697, 0.6819923188761863, 0.6823765804767673, 0.6826419912313295]
# [0.5135172731007596, 0.5179385869562528, 0.5207203984828654, 0.5223345129079338, 0.5232990928437897, 0.5239042976299548, 0.5243012323989288, 0.5245705339598379, 0.5247576054229797, 0.5248895949936887]
# [0.5943596950574914, 0.6097531889168286, 0.6179180528991234, 0.6222635694309935, 0.6245904327755052, 0.6258436427598952, 0.6265218428403796, 0.6268902645954744, 0.6270910145045151, 0.6272006738111018]


#add cpds of t_1
# mode = 'none'
# mode = "gaussian"
# dbn.add_cpds(tabular_cpd_t_1)
# bn.add_cpds(tabular_cpd_t_1_bn)



# dat = bn.simulate(int(1e4))
# # print(bn.nodes)
# # print(data_bn.columns)
# # bmp = BayesianModelProbability(bn)
# # likelihood = bmp.score(data_bn)

# likelihood = log_likelihood_score(bn, data_bn)
# print(likelihood)

# def print_full(cpd):
#     backup = TabularCPD._truncate_strtable
#     TabularCPD._truncate_strtable = lambda self, x: x
#     print(cpd)
#     TabularCPD._truncate_strtable = backup

# Access the CPDs
# for node in dbn.nodes():
#     cpd = dbn.get_cpds(node)
#     # cpd_df = pd.DataFrame(cpd.values, columns=cpd.variables, index=cpd.variables)
#     print(f"CPD of {node}: {cpd}")
# print()












# from pgmpy.estimators import BayesianEstimator
# from pgmpy.factors.discrete import Dirichlet
# from pgmpy.models import BayesianModel

# model = BayesianModel([('A', 'B'), ('A', 'C')])
# prior = {'A': Dirichlet([1, 2]), 'B': None, 'C': None}
# estimator = BayesianEstimator(model, data, prior=prior)
# cpd_B = estimator.estimate_cpd('B')
# cpd_C = estimator.estimate_cpd('C')
#
#
# from pgmpy.estimators import BayesianEstimator
# from pgmpy.models import BayesianModel
# from pgmpy.factors.discrete import TabularCPD
#
# model = BayesianModel([('A', 'B'), ('A', 'C')])
# cpd_A = TabularCPD('A', 2, [[0.3, 0.7]])
# prior = {'A': cpd_A, 'B': None, 'C': None}
# estimator = BayesianEstimator(model, data, prior=prior)
# cpd_B = estimator.estimate_cpd('B')
# cpd_C = estimator.estimate_cpd('C')

