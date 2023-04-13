import pandas as pd
import numpy as np
from hyppo.ksample import MMD
from scipy import stats
from scipy.stats import anderson_ksamp

chat_id = 1066531890 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    alpha = 0.05
    #p_value = MMD(compute_kernel = "rbf", gamma = 0.5).test(x, y)[1]
    #p_value = stats.cramervonmises_2samp(x, y).pvalue
    p_value = anderson_ksamp([x, y]).pvalue 
    result = p_value < alpha
    return result
