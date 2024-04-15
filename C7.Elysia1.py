import numpy as np

# 行列集合の生成
matrix_set = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]

# 任意の二項演算の関数
def operation(A, B):
    return np.sum(A * B)  # 行列の要素ごとの積の総和を採用

# 左辺を計算する関数
def left_side(matrix_set, operation):
    max_value = None
    for A in matrix_set:
        min_result = None
        for B in matrix_set:
            result = operation(A, B)
            if min_result is None or result < min_result:
                min_result = result
        if max_value is None or min_result > max_value:
            max_value = min_result
    return max_value

# 右辺を計算する関数
def right_side(matrix_set, operation):
    min_value = None
    for B in matrix_set:
        max_result = None
        for A in matrix_set:
            result = operation(A, B)
            if max_result is None or result > max_result:
                max_result = result
        if min_value is None or max_result < min_value:
            min_value = max_result
    return min_value

# 左辺と右辺が等しいことを確認
left_result = left_side(matrix_set, operation)
right_result = right_side(matrix_set, operation)

if np.allclose(left_result, right_result):
    print("左辺と右辺が等しい：", left_result)
else:
    print("左辺と右辺が等しくない")



import numpy as np

# 目的関数
def objective_function(A, B):
    return np.sum(A), -np.sum(B)  # Bを最小化するために符号を反転

# 制約条件
def constraint(A, B):
    return np.sum(A) <= 10

# 行列の集合
matrix_set = [np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]]), np.array([[3, 4], [5, 6]])]

# 目的関数と制約条件を満たす最適な A と B を見つける
best_A = None
best_B = None
max_objective_value_A = -np.inf
min_objective_value_B = np.inf

for A in matrix_set:
    for B in matrix_set:
        # 制約条件を満たすかどうかを確認
        if constraint(A, B):
            # 目的関数を計算
            objective_value_A, objective_value_B = objective_function(A, B)
            # 目的関数を最大化する A と最小化する B を更新
            if objective_value_A > max_objective_value_A:
                max_objective_value_A = objective_value_A
                best_A = A
                best_B = B

print("最適な A:", best_A)
print("最適な B:", best_B)



import numpy as np

# 目的関数
def objective_function(A, B):
    return np.sum(A) + np.sum(B)

# 制約条件
def constraint_1(A, B):
    return np.sum(A) + np.sum(B) <= 50

def constraint_2(A, B):
    return np.sum(A @ B) >= 100

# 行列の集合
matrix_set = [np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]]), np.array([[3, 4], [5, 6]])]

# 最適な A と B を見つける
best_A = None
best_B = None
max_objective_value = -np.inf

for A in matrix_set:
    for B in matrix_set:
        # 制約条件を満たすかどうかを確認
        if constraint_1(A, B) and constraint_2(A, B):
            # 目的関数を計算
            objective_value = objective_function(A, B)
            # 目的関数を最大化する A と B を更新
            if objective_value > max_objective_value:
                max_objective_value = objective_value
                best_A = A
                best_B = B

print("最適な A:", best_A)
print("最適な B:", best_B)
print("目的関数の値:", max_objective_value)


