import pandas as pd
import numpy as np
from hyppo.ksample import MMD


chat_id = 1066531890 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    alpha = 0.05
    p_value = MMD(compute_kernel = "rbf", gamma = 1).test(x, y)[1]
    result = p_value < alpha
    return result
