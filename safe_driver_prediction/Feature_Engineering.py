from data_util import *
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

train_data = load_train_data()

#Split the columns to identify data types
columns_name = [x.split("_") for x in train_data.columns.tolist()]

#Make subgroups of data
cat_dat = train_data.iloc[:, list(np.where(['cat' in x for x in columns_name])[0])]
bin_dat = train_data.iloc[:, list(np.where(['bin' in x for x in columns_name])[0])]

cont_dis_dat = train_data.iloc[:, list(np.where([not ('cat' in x or 'bin' in x) for x in columns_name])[0])[2:]]

na_numbers = (train_data == -1).sum(axis=1)

#Deal with the missing data
def handle_missing_data(dat,method):
    #This method is only for cont_dis_data
    if method == "mean_mode":
        cont_dat = dat.iloc[:, [4, 5, 6, 8, 9, 10, 11, 12, 13, 14]]
        dis_dat = dat.iloc[:, [x not in [4, 5, 6, 8, 9, 10, 11, 12, 13, 14] for x in range(0, dat.columns.__len__())]]
        index_mean = cont_dat.apply(np.mean, "index")
        index_mode = dis_dat.apply(stats.mode, "index")

        for i in range(cont_dat.columns.__len__()):
            cont_dat.iloc[cont_dat.iloc[:,i] == -1, i] = index_mean[i]

        for i in range(dis_dat.columns.__len__()):
            dis_dat.iloc[dis_dat.iloc[:, i] == -1, i] = index_mode[i]

        return pd.concat([cont_dat, dis_dat], axis =1)

    # This method is for train_dat
    if method == "delete":
        dat.apply(lambda x: sum(x == -1), "index")

    if method == "binary":
        0

    if method == "new_column":
        0

    if method == "new_factor":
        0

#Create new features(the NA numbers and the sum of binary variables)
bin_sum = bin_dat.sum(axis=1)

generate_data = pd.DataFrame({'na_num': na_numbers, 'bin_sum': bin_sum})


#Standardize continues and discrete variables
cont_dis_dat = pd.DataFrame(StandardScaler().fit_transform(cont_dis_dat), columns=cont_dis_dat.columns)


#Transfer catagorical variables to dummy variables
cat_dat = OneHotEncoder().fit_transform(cat_dat)







