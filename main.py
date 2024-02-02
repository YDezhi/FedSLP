import numpy as np
import main_fed_hetero

for iid in [3]:
    for flag in [5]:
        main_fed_hetero.mainfunc(flag, iid)


# 点火率每一个tensor是一层 每个tensor里面的一个数据是一个时间步

