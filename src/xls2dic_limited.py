import pandas as pd
# import pyexcel
# import pyexcel.ext.xls


file_path = r'/home/zzahedi/project-wellbeing/survey_df.xlsx'
file_path_export = r'/home/zzahedi/project-wellbeing/exp_df_limit.xlsx'
file_path_export2 = r'/home/zzahedi/project-wellbeing/exp2_df_limit.xlsx'
file_path_bin = r'/home/zzahedi/project-wellbeing/bin_df.xlsx'
# xls = pd.ExcelFile(file_path)
# df = xls.parse(xls.sheet_names[0])
df = pd.read_excel(file_path, encoding='utf-16')
# df = pd.read_csv(file_path)
positive_rel = {}
cooperation = {}
drivetype = {}
# for i in range(1, len(df.index)):
#     row = df.iloc[i]
print(df)
for index, row in df.iterrows():
    drivetype[row[0]] = row[3]
    positive_rel[row[0]] = row[29]
    cooperation[row[0]] =row[30]

print(positive_rel, "\n", cooperation, "\n", drivetype)
# dicti = df.to_dict()
# print(df)
pos_rel_val_alt = []
coop_val_alt = []
pos_rel_val_self = []
coop_val_self = []

for i in drivetype:
    if drivetype[i][3:6] == '310' or drivetype[i][3:6] == "510":
        pos_rel_val_alt.append(positive_rel[i])
        coop_val_alt.append(cooperation[i])
    if drivetype[i][3:6] == '311' or drivetype[i][3:6] == "511":
        pos_rel_val_self.append(positive_rel[i])
        coop_val_self.append(cooperation[i])


coop_bin_alt = []
coop_bin_self = []
for j in coop_val_alt:
    if j >= 50:
        coop_bin_alt.append(1)
    else:
        coop_bin_alt.append(0)

for j in coop_val_self:
    if j >= 50:
        coop_bin_self.append(1)
    else:
        coop_bin_self.append(0)
# for i in positive_rel:
#     if i[4:6] == '10':
#         pos_rel_val_alt.append(positive_rel[i])
#         coop_val_alt.append(cooperation[i])
#     if i[4:6] == '11':
#         pos_rel_val_self.append(positive_rel[i])
#         coop_val_self.append(cooperation[i])

print(len(pos_rel_val_alt), pos_rel_val_alt, "\n", len(coop_val_alt), coop_val_alt, "\n", len(pos_rel_val_self), pos_rel_val_self, "\n", len(coop_val_self), coop_val_self)
df_pos_alt = pd.DataFrame({'pos_alt': pos_rel_val_alt, 'coop_alt': coop_val_alt, 'coop_alt_bin': coop_bin_alt})
df_self = pd.DataFrame({'pos_self': pos_rel_val_self, 'coop_self': coop_val_self, 'coop_self_bin': coop_bin_self})
# df_bin = pd.DataFrame({'coop_alt_bin': coop_bin_alt}, {'coop_self_bin': coop_bin_self})
# df_coop_alt = pd.DataFrame(coop_val_alt)
# df_pos_self = pd.DataFrame(pos_rel_val_self)
# df_coop_self = pd.DataFrame(coop_val_self)
df_pos_alt.to_excel(file_path_export)
df_self.to_excel(file_path_export2)
# df_bin.to_excel(file_path_bin)
# pyexcel.save_as(array = pos_rel_val_alt, dest_file_name = "pos_alt.xls")