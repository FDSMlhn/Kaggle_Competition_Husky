import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#Create new features(the NA numbers and the sum of binary variables)
def creat_new_features(bd,td):
    bin_sum = (bd > 0).sum(axis=1)
    na_numbers = (td == -1).sum(axis=1)

    return pd.DataFrame({'na_num': na_numbers, 'bin_sum': bin_sum})


#Deal with the missing data
def handle_missing_data(dat, method, threshold=0.1, replace_number=6666):
    #This method is only for cont_dis_data
    if method == "mean_mode":
        cont_dat = dat.iloc[:, [4, 5, 6, 8, 9, 10, 11, 12, 13, 14]]
        dis_dat = dat.iloc[:, [x not in [4, 5, 6, 8, 9, 10, 11, 12, 13, 14] for x in range(0, dat.columns.__len__())]]
        index_mean = cont_dat.apply(lambda x: np.mean(x[x != -1]), "index")
        index_mode = dis_dat.apply(lambda x: stats.mode(x[x != -1]), "index")

        for i in range(cont_dat.columns.__len__()):
            cont_dat.iloc[:, i].replace(-1, index_mean[i])

        for i in range(dis_dat.columns.__len__()):
            dis_dat.iloc[:, i].replace(-1, index_mode[i][0][0])

        return pd.concat([cont_dat, dis_dat], axis=1)

    # This method is for train_dat
    if method == "delete":
        missing_value_feature = dat.apply(lambda x: sum(x == -1), "index")
        dat = dat.iloc[:, np.where(missing_value_feature / dat.iloc[:, 0].__len__() < threshold)[0].tolist()]

        return dat

    # This method is only for cont_dis_data
    if method == "binary":
        missing_value_feature = dat.apply(lambda x: sum(x == -1), "index")
        dat.iloc[:, np.where(missing_value_feature / dat.iloc[:, 0].__len__() >= threshold)[0].tolist()] = (dat.iloc[:, np.where(missing_value_feature / dat.iloc[:, 0].__len__() >= threshold)[0].tolist()] != -1).astype(int)

        return dat

    # This method is only for binary and cat_data
    if method == "new_factor":
        dat = dat.replace(-1, replace_number)

        return dat


def feature_main(dat):
    # Split the columns to identify data types
    columns_name = [x.split("_") for x in dat.columns.tolist()]

    # Make subgroups of data
    cat_dat = dat.iloc[:, list(np.where(['cat' in x for x in columns_name])[0])]
    bin_dat = dat.iloc[:, list(np.where(['bin' in x for x in columns_name])[0])]

    cont_dis_dat = dat.iloc[:, list(np.where([not ('cat' in x or 'bin' in x) for x in columns_name])[0])[2:]]

    new_features = creat_new_features(bin_dat,dat)

    cont_dis_dat = handle_missing_data(cont_dis_dat, "mean_mode")

    bin_dat = handle_missing_data(bin_dat, "new_factor")

    cat_dat = handle_missing_data(cat_dat, "new_factor")

    # Standardize continues and discrete variables
    cont_dis_dat = pd.DataFrame(StandardScaler().fit_transform(cont_dis_dat), columns=cont_dis_dat.columns)

    # Transfer catagorical variables to dummy variables
    #col = cat_dat.columns
    cat_dat = OneHotEncoder().fit_transform(cat_dat)

    test = pd.concat([dat.iloc[:, [0, 1]], cont_dis_dat, bin_dat, new_features, pd.DataFrame(cat_dat.toarray())], axis=1)

    return test







