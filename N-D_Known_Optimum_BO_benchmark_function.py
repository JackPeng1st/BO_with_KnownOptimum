import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import  numpy as np
 
# total iteration
iter = 50
# of init point 
num_init = 10
# is a maximization problem ? yes：1, no: -1
is_max = -1
# black box function name
name_f = 'Drop Wave function'
#black box function：input is a list
def f(var_value_list,is_max):
    b=0.5*(var_value_list[0]*var_value_list[0]+var_value_list[1]*var_value_list[1])+2
    a=-(1+np.cos(12*np.sqrt(var_value_list[0]*var_value_list[0]+var_value_list[1]*var_value_list[1])))/b   
    return a * is_max
# bounds of vars
bound_list = [[-5.12,5.12],[-5.12,5.12]]
# optimum value
fstar = -1 * is_max

'''
# visualization
x1=np.linspace(-5.12,5.12,100)
x2=np.linspace(-5.12,5.12,100)
X1,X2=np.meshgrid(x1,x2)
 
def plotter(E,A):
    fig=plt.figure(figsize=[12,8])
    ax=plt.axes(projection='3d')
    ax.plot_surface(X1,X2,f(X1,X2),color='red',alpha=0.7)
    ax.plot_wireframe(X1,X2,f(X1,X2),ccount=2,rcount=2,color='pink', alpha=0.2)
    ax.view_init(elev=E,azim=A)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    plt.show()
 
from ipywidgets import interactive
iplot=interactive(plotter,E=(-60,30,5),A=(-60,30,5))
iplot
'''
import matplotlib.pyplot as plt
from bayes_opt import BayesOpt_KnownOptimumValue,BayesOpt
import numpy as np
from bayes_opt import vis_ERM,functions
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

# initial 
import random 
random.seed( 999 )
init_x_list = []
init_y_list = []
for i in range(num_init):
    # generate values of X1 to Xn
    var_value_list = []
    for var_i in range(len(bound_list)):
        value = random.uniform(bound_list[var_i][0],bound_list[var_i][1])
        var_value_list.append(value)

    y = f(var_value_list,is_max) 
    init_x_list.append(var_value_list)
    init_y_list.append(y)


# ### Random Search

x_list = init_x_list.copy()
y_list = init_y_list.copy()

for i in range(iter):
    # generate values of X1 to Xn
    var_value_list = []
    for var_i in range(len(bound_list)):
        value = random.uniform(bound_list[var_i][0],bound_list[var_i][1])
        var_value_list.append(value)
    y = f(var_value_list,is_max)
    x_list.append(var_value_list)
    y_list.append(y)
    
def max_index(y_list):
    index_list = []
    for i in range(len(y_list)):
        y = y_list[i]
        max_value = max(y_list)
        if y == max_value:
            index_list.append(i)
    print(max_value)
    return index_list

index_list = max_index(y_list)

def best_x_value(x_list,index_list):
    best_x_value_list = []
    for i in index_list:
        best_x_value_list.append(x_list[i])
    return best_x_value_list
best_x_value(x_list,index_list)

random_list = y_list 


# Black box function for BO
class YourFunction:
    def __init__(self):
       
        # define the search range for each variable
        self.bounds = np.asarray(bound_list)
            
        self.input_dim = self.bounds.shape[0] 

        # do we want to maximize the function or minimize ?
        #self.ismax= is_max  # set -1 if we want to minimize
        
        # define the known optimum value if it is available
        self.fstar = fstar
        
        # define the name of your function
        self.name = name_f
        
    def evaluate_single_fx(self,X): 
        # evaluate y=f(X)
        X = np.reshape(X,self.input_dim)
        X = list(X)
      
        return f(X,is_max)
   
    
    def func(self,X):
        X=np.asarray(X)        
       
        if len(X.shape)==1: # 1 data point
            fx=self.evaluate_single_fx(X)
        else: # multiple data points
            fx=np.apply_along_axis( self.evaluate_single_fx,1,X)
            
        return fx  
myfunction=YourFunction()

# ### Normal GP with UCB acquisition function
# initial
init_X = np.asarray(init_x_list)
init_Y=myfunction.func(init_X)
print(init_Y)

# create an empty object for BO using transformed GP
acq_name='ucb'
IsTGP=0 # using Transformed GP

bo_tgp=BayesOpt_KnownOptimumValue(myfunction.func,myfunction.bounds,fstar=myfunction.fstar,                               acq_name=acq_name,IsTGP=IsTGP,verbose=1)
bo_tgp.init_with_data(init_X=init_X,init_Y=init_Y)

for index in range(0,iter):
    xt=bo_tgp.select_next_point()
    #print(bo_tgp.X_ori[-1])
    print(tabulate([[ index,np.round(bo_tgp.X_ori[-1],3), np.round(bo_tgp.Y_ori[-1],3), np.round(bo_tgp.Y_ori.max(),3)]],                headers=['Iter','Selected x', 'Output y=f(x)', 'Best Observed Value']))

GP_UCB_list = list(bo_tgp.Y_ori)

# ### Normal GP with EI acquisition function
# initial
init_X = np.asarray(init_x_list)
init_Y=myfunction.func(init_X)
print(init_Y)

# create an empty object for BO using transformed GP
acq_name='ei'
IsTGP=0 # using Transformed GP

bo_tgp=BayesOpt_KnownOptimumValue(myfunction.func,myfunction.bounds,fstar=myfunction.fstar,                               acq_name=acq_name,IsTGP=IsTGP,verbose=1)
bo_tgp.init_with_data(init_X=init_X,init_Y=init_Y)

for index in range(0,iter):
    xt=bo_tgp.select_next_point()
    #print(bo_tgp.X_ori[-1])
    print(tabulate([[ index,np.round(bo_tgp.X_ori[-1],3), np.round(bo_tgp.Y_ori[-1],3), np.round(bo_tgp.Y_ori.max(),3)]],                headers=['Iter','Selected x', 'Output y=f(x)', 'Best Observed Value']))

GP_EI_list = list(bo_tgp.Y_ori)

# ### Transformed GP with UCB acquisition function

init_X = np.asarray( init_x_list)
init_Y=myfunction.func(init_X)
print(init_Y)

# create an empty object for BO using transformed GP
acq_name='ucb'
IsTGP=1 # using Transformed GP

bo_tgp=BayesOpt_KnownOptimumValue(myfunction.func,myfunction.bounds,fstar=myfunction.fstar,                               acq_name=acq_name,IsTGP=IsTGP,verbose=1)
bo_tgp.init_with_data(init_X=init_X,init_Y=init_Y)

for index in range(0,iter):
    xt=bo_tgp.select_next_point()
    #print(bo_tgp.X_ori[-1])
    print(tabulate([[ index,np.round(bo_tgp.X_ori[-1],3), np.round(bo_tgp.Y_ori[-1],3), np.round(bo_tgp.Y_ori.max(),3)]],                headers=['Iter','Selected x', 'Output y=f(x)', 'Best Observed Value']))

TGP_UCB_list = list(bo_tgp.Y_ori)


# ### Transformed GP with EI acquisition function

init_X = np.asarray( init_x_list)
init_Y=myfunction.func(init_X)
print(init_Y)

# create an empty object for BO using transformed GP
acq_name='ei'
IsTGP=1 # using Transformed GP

bo_tgp=BayesOpt_KnownOptimumValue(myfunction.func,myfunction.bounds,fstar=myfunction.fstar,                               acq_name=acq_name,IsTGP=IsTGP,verbose=1)
bo_tgp.init_with_data(init_X=init_X,init_Y=init_Y)

for index in range(0,iter):
    xt=bo_tgp.select_next_point()
    #print(bo_tgp.X_ori[-1])
    print(tabulate([[ index,np.round(bo_tgp.X_ori[-1],3), np.round(bo_tgp.Y_ori[-1],3), np.round(bo_tgp.Y_ori.max(),3)]],                headers=['Iter','Selected x', 'Output y=f(x)', 'Best Observed Value']))

TGP_EI_list = list(bo_tgp.Y_ori)

# ### Transformed GP with CBM acquisition function
# 

init_X = np.asarray(init_x_list)
init_Y=myfunction.func(init_X)
print(init_Y)

# create an empty object for BO using transformed GP
acq_name='cbm'
IsTGP=1 # using Transformed GP

bo_tgp=BayesOpt_KnownOptimumValue(myfunction.func,myfunction.bounds,fstar=myfunction.fstar,                               acq_name=acq_name,IsTGP=IsTGP,verbose=1)
bo_tgp.init_with_data(init_X=init_X,init_Y=init_Y)

for index in range(0,iter):
    xt=bo_tgp.select_next_point()
    #print(bo_tgp.X_ori[-1])
    print(tabulate([[ index,np.round(bo_tgp.X_ori[-1],3), np.round(bo_tgp.Y_ori[-1],3), np.round(bo_tgp.Y_ori.max(),3)]],                headers=['Iter','Selected x', 'Output y=f(x)', 'Best Observed Value'])) 

TGP_CBM_list = list(bo_tgp.Y_ori)



# ### Transformed GP with erm acquisition function
# 

init_X = np.asarray(init_x_list)
init_Y=myfunction.func(init_X)
print(init_Y)

# create an empty object for BO using transformed GP
acq_name='erm'
IsTGP=1 # using Transformed GP

bo_tgp=BayesOpt_KnownOptimumValue(myfunction.func,myfunction.bounds,fstar=myfunction.fstar,                               acq_name=acq_name,IsTGP=IsTGP,verbose=1)
bo_tgp.init_with_data(init_X=init_X,init_Y=init_Y)

for index in range(0,iter):
    xt=bo_tgp.select_next_point()
    #print(bo_tgp.X_ori[-1])
    print(tabulate([[ index,np.round(bo_tgp.X_ori[-1],3), np.round(bo_tgp.Y_ori[-1],3), np.round(bo_tgp.Y_ori.max(),3)]],                headers=['Iter','Selected x', 'Output y=f(x)', 'Best Observed Value'])) 

TGP_ERM_list = list(bo_tgp.Y_ori)


# ### Comparison

# 輸出當前iteration最佳值
def cur_best_list (num_list):
    new_list = []
    best = num_list[0]
    for num in num_list:
        if(num > best):
            new_list.append(num)
            best = num
        else:
            new_list.append(best)
    return new_list

random_list = cur_best_list(random_list)
GP_UCB_list = cur_best_list(GP_UCB_list)
GP_EI_list = cur_best_list(GP_EI_list)
TGP_UCB_list = cur_best_list(TGP_UCB_list)
TGP_EI_list = cur_best_list(TGP_EI_list)
TGP_CBM_list = cur_best_list(TGP_CBM_list)
TGP_ERM_list = cur_best_list(TGP_ERM_list)

x = list(range(iter))
plt.plot(x,random_list[num_init:], color='grey',label='random')
plt.plot(x,GP_UCB_list[num_init:], color='lightcoral',label='GP-UCB')
plt.plot(x,GP_EI_list[num_init:], color='y',label='GP-EI')
plt.plot(x,TGP_UCB_list[num_init:], color='palegreen',label='TGP-UCB')
plt.plot(x,TGP_EI_list[num_init:], color='skyblue',label='TGP-EI')
plt.plot(x,TGP_CBM_list[num_init:],color='deeppink',label='TGP-CBM')
plt.plot(x,TGP_ERM_list[num_init:],color='purple',label='TGP-ERM')
plt.title(name_f)
plt.legend(loc = 'upper left')
plt.show()