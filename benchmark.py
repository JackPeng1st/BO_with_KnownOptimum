import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class Drop_Wave:
    def __init__(self):
        self.bounds = [[-5.12,5.12],[-5.12,5.12]]    
        self.ismax=-1
        self.fstar = -1 * self.ismax       
        self.name = 'Drop Wave function'
    def func(self,var_value_list,is_max):
        b=0.5*(var_value_list[0]*var_value_list[0]+var_value_list[1]*var_value_list[1])+2
        a=-(1+np.cos(12*np.sqrt(var_value_list[0]*var_value_list[0]+var_value_list[1]*var_value_list[1])))/b   
        return a * is_max
    def bound_list(self):
        return self.bounds
    def name_f(self):
        return self.name
    def is_max(self):
        return self.ismax
    def f_star(self):
        return self.fstar

class Ackley:
    def __init__(self):
        self.bounds = [[-32.768, 32.768],[-32.768, 32.768]]    
        self.ismax=-1
        self.fstar = 0 * self.ismax       
        self.name = 'Ackley function'
    def func(self,var_value_list,is_max):
        num = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (var_value_list[0]**2 + var_value_list[1]**2)))-np.exp(0.5 * (np.cos(2 * np.pi * var_value_list[0])+np.cos(2 * np.pi * var_value_list[1]))) + np.e + 20   
        return num * is_max
    def bound_list(self):
        return self.bounds
    def name_f(self):
        return self.name
    def is_max(self):
        return self.ismax
    def f_star(self):
        return self.fstar

class Eggholder:
    def __init__(self):
        self.bounds = [[-512,512],[-512,512]]    
        self.ismax=-1
        self.fstar = -959.6407 * self.ismax       
        self.name = 'Eggholder function'
    def func(self,var_value_list,is_max):
        a=np.sqrt(np.fabs(var_value_list[1]+var_value_list[0]/2+47))
        b=np.sqrt(np.fabs(var_value_list[0]-(var_value_list[1]+47)))
        c=-(var_value_list[1]+47)*np.sin(a)-var_value_list[0]*np.sin(b)
        return c * is_max
    def bound_list(self):
        return self.bounds
    def name_f(self):
        return self.name
    def is_max(self):
        return self.ismax
    def f_star(self):
        return self.fstar


class Hartmann_6D:
    def __init__(self):
        self.bounds = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]    
        self.ismax=-1
        self.fstar = -3.32237 * self.ismax       
        self.name = 'Hartmann 6D function'
    def func(self,var_value_list,is_max):
        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A=np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]]

        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;

        fval  =np.zeros((1,1))  
        outer = 0;
        for ii in range(4):
            inner = 0;
            for jj in range(6):
                xj = var_value_list[jj]
                Aij = A[ii,jj]
                Pij = P[ii,jj]
                inner = inner + Aij*(xj-Pij)**2
            
            new = alpha[ii] * np.exp(-inner)
            outer = outer + new

        fval[0][0] = -(2.58 + outer) / 1.94;
        return self.ismax*(fval[0][0])
    def bound_list(self):
        return self.bounds
    def name_f(self):
        return self.name
    def is_max(self):
        return self.ismax
    def f_star(self):
        return self.fstar

class xgb_hyper_iris:
    
    def __init__(self):
        self.bounds = [[3.0,9.0],  # max_depth
                        [0.1,0.5], # eta
                        [1,20], #min_child_weight
                        [0.1,1]] # colsample_bytree   
        self.ismax= 1
        self.fstar = 1 * self.ismax       
        self.name = 'xgb_hyper_iris'
    def func(self,var_value_list,is_max):
        
        x1,x2,x3,x4=var_value_list[0],var_value_list[1],var_value_list[2],var_value_list[3]
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=999)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        param = {
        'max_depth': round(x1),  # the maximum depth of each tree
        'eta': x2,  # the training step for each iteration
        'min_child_weight':x3,
        'colsample_bytree':x4,
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 3}  # the number of classes that exist in this datset
        model = xgb.train(param, dtrain)
        preds = model.predict(dtest)
        preds = np.asarray([np.argmax(line) for line in preds])
        y = accuracy_score(y_test, preds)
        return self.ismax*y

    def bound_list(self):
        return self.bounds
    def name_f(self):
        return self.name
    def is_max(self):
        return self.ismax
    def f_star(self):
        return self.fstar