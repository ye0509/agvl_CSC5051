import numpy as np
from scipy import stats

def voc_calculation(array1, array2):

    # 计算Spearman秩相关系数
    spearman_corr, spearman_p = stats.spearmanr(array1, array2)

    # 计算Kendall秩相关系数
    kendall_corr, kendall_p = stats.kendalltau(array1, array2)

    return spearman_corr, kendall_corr
