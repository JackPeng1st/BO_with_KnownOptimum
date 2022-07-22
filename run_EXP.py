from  EXP_Known_Opt_BO import exp
from  benchmark import Drop_Wave, Ackley, Eggholder,Hartmann_6D,xgb_hyper_iris
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# # of Replication
repli = 5

# total iteration
iter_list = [20,30,50]
# of init point 
num_init_list = [5,10,20]

f = xgb_hyper_iris()
is_max = f.is_max()
name_f = f.name_f()
bound_list = f.bound_list()
fstar = f.f_star()

#exp(repli,iter[0],num_init[0],is_max,name_f,bound_list,fstar,f.func)

for i in range(len(iter_list)): 
    exp(repli,iter_list[i],num_init_list[i],is_max,name_f,bound_list,fstar,f.func)


