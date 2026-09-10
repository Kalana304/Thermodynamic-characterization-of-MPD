import os
import numpy as np

def get_parity_check_alist(alist_path):
    alist_string = open(alist_path).readlines()
    alist = []

    for ele in alist_string:
        temp = list(map(int, ele.split()))
        alist.append(temp)

    n = alist[0][0]
    m = alist[0][1]
    v_max = alist[1][0]
    c_max = alist[1][1]

    VariableDegree = alist[2]
    CheckDegree = alist[3]

    assert sum(VariableDegree)==sum(CheckDegree)
    assert max(VariableDegree)==v_max
    assert max(CheckDegree)==c_max

    var_connect = alist[4 : n + 3]
    chk_connect = alist[n + 3 : m + n + 3] 

    print(f"var nodes = {n}")
    print(f"check nodes = {m}")

    H = np.zeros((m, n), dtype=int)
    for j in range(m):
        for i in chk_connect[j]:
            H[j, i - 1] = 1
    
    return n, m, v_max, c_max, chk_connect, var_connect, H