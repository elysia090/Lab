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
