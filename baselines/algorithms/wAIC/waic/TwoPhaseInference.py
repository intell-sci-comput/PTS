import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize 
from sklearn.linear_model import LassoLarsCV 
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error 
from math import log 
import random
import warnings
warnings.filterwarnings("ignore")

def calculate_aic(n, mse, num_params):
    aic = n * log(mse) + 2 * num_params 
    return aic

def library_matching(df1, df2):
    new_data = pd.DataFrame()
    for ii in range(len(df1._stat_axis.values.tolist())):
        if df1._stat_axis.values.tolist()[ii] != 'constant':
            index_name = df1._stat_axis.values.tolist()[ii]
            tmp = df2[index_name]
            new_data = pd.concat([new_data,tmp],axis=1)
    return new_data


def terms_sort_Lars(X_lib, Y_goal, intercept):
    """
    Perform Lasso Lars regression and sort terms.

    Args:
    X_lib (pd.DataFrame): Input features
    Y_goal (pd.Series): Target variable
    intercept (bool): Whether to fit intercept

    Returns:
    tuple: (Score, MSE, AIC, sorted coefficients)
        - Score (float): R-squared score of the model
        - MSE (float): Mean squared error of the model
        - AIC (float): Akaike Information Criterion of the model
        - sort (pd.Series): Sorted series of coefficients, where the index is the feature name 
                            and the value is the corresponding coefficient. Sorted in ascending order.

    Example:
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> y = pd.Series([7, 8, 9])
    >>> score, mse, aic, sort = terms_sort_Lars(X, y, True)
    >>> print(f"Score: {score:.2f}, MSE: {mse:.2f}, AIC: {aic:.2f}")
    Score: 1.00, MSE: 0.00, AIC: -inf
    >>> print(sort)
    A    1.0
    B    1.0
    constant    4.0
    dtype: float64
    """
    # reg = LassoLarsCV(cv=5, fit_intercept=intercept,  n_jobs=-1, max_iter=1000, normalize=False).fit(X_lib,Y_goal)
    reg = LassoLarsCV(cv=5, fit_intercept=intercept,  n_jobs=-1, max_iter=1000).fit(X_lib,Y_goal)
    coef = pd.Series(reg.coef_, index=X_lib.columns)
    if intercept == True:
        coef['constant'] = reg.intercept_
        num_params = len(coef)
    else:
        num_params = len(coef)    
    P = X_lib
    Score = reg.score(X_lib,Y_goal)
    yhat = reg.predict(P)
    mse = mean_squared_error(Y_goal, yhat)
    aic = calculate_aic(len(Y_goal), mse, num_params)
    sort = coef.sort_values()
    return Score, mse, aic, sort

def terms_sort_Lasso(X_lib,Y_goal,intercept):
    """
    Perform Lasso regression and sort terms.

    Args:
    X_lib (pd.DataFrame): Input features
    Y_goal (pd.Series): Target variable
    intercept (bool): Whether to fit intercept

    Returns:
    tuple: (Score, MSE, AIC, sorted coefficients)

    Example:
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> y = pd.Series([7, 8, 9])
    >>> score, mse, aic, sort = terms_sort_Lasso(X, y, True)
    >>> print(f"Score: {score:.2f}, MSE: {mse:.2f}, AIC: {aic:.2f}")
    Score: 1.00, MSE: 0.00, AIC: -inf
    """

    # reg = LassoCV(cv=5, fit_intercept=intercept, n_jobs=-1, max_iter=1000, normalize=False).fit(X_lib,Y_goal)
    reg = LassoCV(cv=5, fit_intercept=intercept, n_jobs=-1, max_iter=1000).fit(X_lib,Y_goal)
    coef = pd.Series(reg.coef_, index=X_lib.columns)
    if intercept == True:
        coef['constant'] = reg.intercept_
        num_params = len(coef)
    else:
        num_params = len(coef)    
    P = X_lib
    Score = reg.score(X_lib,Y_goal)
    yhat = reg.predict(P)
    mse = mean_squared_error(Y_goal, yhat)
    aic = calculate_aic(len(Y_goal), mse, num_params)
    #print('label of function: %.3f' % time)
    sort = coef.sort_values()
    return Score, mse, aic, sort

def random_batch_generation(X, y, Number, batch_number):
    Xlen = X.iloc[:,0].size/Number
    List = range(0,Number)
    sample_list = random.sample(List,batch_number)
    X_sample = pd.DataFrame()
    y_sample = pd.DataFrame()
    for i in range(0,batch_number):
        start = int(sample_list[i]*Xlen)
        end = int((sample_list[i]+1)*Xlen)
        tmp1 = X.iloc[start:end,:]
        tmp2 = y.iloc[start:end,:]
        X_sample = X_sample.append(tmp1)
        y_sample = y_sample.append(tmp2)
    return X_sample, y_sample, sample_list

def PhaseOne(FuncMatrix, NumDiv, Nnodes, dim, Dim, keep, Lambda, plotstart, plotend):
    X_all = FuncMatrix*1
    y_all = NumDiv*1
    Xlen = X_all.iloc[:,0].size/Nnodes
    T_len = int(Xlen)
    Lambda = Lambda.values
    if Dim>=3:
        X = X_all.iloc[T_len*0:T_len*20,:]
        y = y_all.iloc[T_len*0:T_len*20,:]
    else:
        X = X_all
        y = y_all

    X_mat = X.values 
    y_mat = y.values
    x_norml1 = []
    y_norml1 = []
    num = len(X_mat[0])
    num2 = len(y_mat[0])
    L = len(X_mat)

    for i in range(0,num):
        x_norml1.append(sum(abs(X_mat[:,i])))

    for i in range(0,num2):
        y_norml1.append(sum(abs(y_mat[:,i])))

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    X[X.columns] = normalize(X[X.columns], norm='l1', axis=0)*L
    y[y.columns] = normalize(y[y.columns], norm='l1', axis=0)*L

    X_col = X.columns 
    Xin = X.iloc[:,:]
    out = np.array(y)
    print('out.shape',out.shape)
    y1 = (out[:,dim])

    # reg1 = LassoCV(cv=5, fit_intercept=True, n_jobs=-1, max_iter=10000, normalize=False).fit(Xin, y1)
    reg1 = LassoCV(cv=5, fit_intercept=True, n_jobs=-1, max_iter=10000).fit(Xin, y1)
    print(reg1.score(Xin,y1))
    print('Best threshold: %.3f' % reg1.alpha_)
    for i in range(len(reg1.coef_)):
        reg1.coef_[i] = reg1.coef_[i]*y_norml1[dim]/x_norml1[i]

    coef1 = pd.Series(reg1.coef_, index = X_col)
    imp_ = pd.concat([coef1.sort_values().head(int(keep/2)),
                    coef1.sort_values().tail(int(keep/2))])
    imp_no_cons = imp_ + (1e-10)
    print("Elementary functions discovered by Phase 1 without constant.")
    print(imp_no_cons)

    imp_coef1 = pd.concat([coef1.sort_values().head(int(keep/2)),
                        coef1.sort_values().tail(int(keep/2-1))])
    imp_cons = imp_coef1 + (1e-10)
    imp_cons['constant'] = reg1.intercept_*y_norml1[dim]/L
    print("Elementary functions discovered by Phase 1 with constant.")
    print(imp_cons)

    # Constant test
    imp_coef1_1 = pd.concat([coef1.sort_values().head(int(keep/2)),
                        coef1.sort_values().tail(int(keep/2-2))])
    imp_cons_1  = imp_coef1_1 + (1e-10)
    imp_cons_1['constant'] = reg1.intercept_*y_norml1[dim]/L

    X = FuncMatrix*1
    y = NumDiv*1

    X_test = X.iloc[T_len*0:T_len*10,:]
    y_test = y.iloc[T_len*0:T_len*10,:]
    X_ori = X_test
    y_ori = y_test
    
    print('imp_cons')
    print(imp_cons.shape)
    
    new_data1 = library_matching(imp_cons,X_ori)
    d1 = y_ori.iloc[:,dim]
    
    print('d1')
    print(d1.shape)
    
    new_data2 = library_matching(imp_cons_1,X_ori)
    
    print('-'*20,'Lasso constant','-'*20)
    
    Score_constant, mse_constant, aic_constant, sort_con = terms_sort_Lasso(new_data1,d1,True)
    
    print('-'*20,'Lasso no_con','-'*20)
    
    Score_no_con, mse_no_con, aic_no_con, sort_no_con = terms_sort_Lasso(new_data1,d1,False)
    
    print('-'*20,'Lasso 1','-'*20)
    
    Score_1, mse_1, aic_1, sort_1 = terms_sort_Lasso(new_data2,d1,True)
    
    print('-'*20)
    
    print(aic_constant, aic_no_con, aic_1)
    con1 = aic_constant/aic_constant
    no_con = aic_no_con/aic_constant
    con2 = aic_1/aic_constant
    print(con1,no_con,con2)
    if aic_constant<=0 and aic_no_con>=0:
        intercept = True
    elif aic_constant<=0 and aic_no_con<0:
        if con1-no_con>Lambda[dim,0] and abs(con1-con2)<=Lambda[dim,1]: 
            intercept = True 
            print("This equation may contain a constant term.")
        else:
            intercept = False
            print("This equation may not contain a constant term.")
    elif aic_constant>0 and aic_no_con<0:
        intercept = False
        print("This equation may not contain a constant term.")
    elif aic_constant>0 and aic_no_con>=0:
        if no_con-con1>Lambda[dim,0] and abs(con1-con2)<=Lambda[dim,1]: 
            intercept = True 
            print("This equation may contain a constant term.")
        else:
            intercept = False
            print("This equation may not contain a constant term.")

    PhaseOne_series = pd.DataFrame()
    
    if intercept == True:
        tmp1 = library_matching(imp_cons,X_ori)
        Constant_series = np.ones(shape=(len(tmp1.values[0,:]),1))
        Constant_series = pd.DataFrame(data = Constant_series, columns = ['constant'])
        With_constant_series = pd.concat([tmp1,Constant_series], axis=1)
        PhaseOne_series = With_constant_series.iloc[int(plotstart*T_len):int(plotend*T_len),:]
    else:
        Without_constant_series = library_matching(imp_no_cons,X_ori)
        PhaseOne_series = Without_constant_series.iloc[int(plotstart*T_len):int(plotend*T_len),:]
    return PhaseOne_series,intercept,imp_cons,imp_no_cons

def PhaseTwo(Matrix, NumDiv, Nnodes, dim, keep, SampleTime, batchsize, Lambda, intercept,imp_cons,imp_no_cons):
    """
    Perform Phase Two of the two-phase inference approach.

    Args:
    Matrix (pd.DataFrame): Input matrix
    NumDiv (pd.DataFrame): Numerical derivatives
    Nnodes (int): Number of nodes
    dim (int): Dimension to analyze
    keep (int): Number of terms to keep
    SampleTime (int): Number of sampling iterations
    batchsize (int): Size of each batch
    Lambda (pd.DataFrame): Lambda values
    intercept (bool): Whether to include intercept
    imp_cons (pd.Series): Important terms with constant
    imp_no_cons (pd.Series): Important terms without constant

    Returns:
    tuple: (InferredResults, aic_final)

    Example:
    >>> Matrix = pd.DataFrame(np.random.rand(100, 10))
    >>> NumDiv = pd.DataFrame(np.random.rand(100, 3))
    >>> Lambda = pd.DataFrame(np.random.rand(3, 3))
    >>> imp_cons = pd.Series(np.random.rand(5))
    >>> imp_no_cons = pd.Series(np.random.rand(5))
    >>> result = PhaseTwo(Matrix, NumDiv, 10, 0, 5, 3, 2, Lambda, True, imp_cons, imp_no_cons)
    >>> print(f"InferredResults shape: {result[0].shape}")
    InferredResults shape: (5, 3)
    """
    
    assert keep >= 3
    
    weight = 10
    Lambda = Lambda.values
    Score_final = np.zeros(shape=(keep,SampleTime))
    mse_final = np.zeros(shape=(keep,SampleTime))
    aic_final = np.zeros(shape=(keep,SampleTime))

    InferredResults = pd.DataFrame()

    for ii in range(0,SampleTime):
        X_batch, y_batch, List = random_batch_generation(Matrix,NumDiv,Nnodes,batchsize)
        X_ori = X_batch
        y_ori = y_batch
        d1 = y_ori.iloc[:,dim]

        new_data1 = library_matching(imp_cons,X_ori)
        Score_constant, mse_constant, aic_constant, sort_con = terms_sort_Lasso(new_data1,d1,True)
        Score_no_con, mse_no_con, aic_no_con, sort_no_con = terms_sort_Lasso(new_data1,d1,False) 
        aic_final[0,ii] = aic_constant
        aic_final[1,ii] = aic_no_con
        Score_final[0,ii] = Score_constant
        Score_final[1,ii] = Score_no_con
        mse_final[0,ii] = mse_constant
        mse_final[1,ii] = mse_no_con
        
        if intercept == True:
            new_data = library_matching(imp_cons,X_ori)
            print(new_data.columns)
            Score = np.zeros(shape=(int(keep)-1,1))
            mse = np.zeros(shape=(int(keep)-1,1))
            AIC = np.zeros(shape=(int(keep)-1,1))
            AIC_ori = np.zeros(shape=(int(keep)-1,1))
            for i in range(0,int(keep)-1):
                
                part = new_data.drop(new_data.columns[[i]],axis=1,inplace=False)
                
                Score_,mse_,aic_,sort_=terms_sort_Lasso(part,d1,intercept)
                Score[i] = Score_
                mse[i] = mse_
                AIC[i] = aic_
                AIC_ori[i] = aic_
                Coef_i = abs(imp_cons[new_data.columns[[i]]].values)*weight
                if AIC[i]>=0:
                    AIC[i] = aic_*Coef_i+(1e-10)
                else:
                    AIC[i] = aic_/Coef_i+(1e-10)
            d_idx = np.argsort(AIC,axis=0)

        else:
            new_data = library_matching(imp_no_cons,X_ori)
            print(new_data.columns)
            Score = np.zeros(shape=(int(keep),1))
            mse = np.zeros(shape=(int(keep),1))
            AIC = np.zeros(shape=(int(keep),1))
            AIC_ori = np.zeros(shape=(int(keep),1))
            for i in range(0,int(keep)):
                
                part = new_data.drop(new_data.columns[[i]], axis=1, inplace=False)
                
                Score_, mse_, aic_, sort_ = terms_sort_Lasso(part,d1,intercept)
                Score[i] = Score_
                mse[i] = mse_
                AIC[i] = aic_
                AIC_ori[i] = aic_
                Coef_i = abs(imp_no_cons[new_data.columns[[i]]].values)*weight
                if AIC[i] >= 0:
                    AIC[i] = aic_*Coef_i+(1e-10)
                else:
                    AIC[i] = aic_/Coef_i+(1e-10)
            d_idx = np.argsort(AIC,axis=0)

        name_idx = []
        functions = locals()
        End = int(keep-2)
        assert keep >= 3
        for i in range(0,End):
            name_idx.append(new_data.columns.values.tolist()[int(d_idx[i,0])])
            new_data_drop=new_data.drop(name_idx,axis=1,inplace=False)
            Score_d, mse_d, aic_d, sort_d = terms_sort_Lasso(new_data_drop,d1,intercept)
            Score_final[i+2,ii] = Score_d
            mse_final[i+2,ii] = mse_d
            aic_final[i+2,ii] = aic_d
            functions['s'+str(i)]=sort_d

        Max = max(aic_final[:,ii])
        Min = min(aic_final[:,ii])
        orinorm_aic = aic_final[:,ii]/(Max-Min)
        print("orinorm_aic:")
        print(orinorm_aic)
        
        stop = 0
        for i in range(2,keep-1):
            if aic_final[i+1,ii]-aic_final[i,ii]<=0:
                stop = stop+1
            #elif aic_final[i+1,ii]-aic_final[i,ii]>=0 and norm_aic[i-2+1]-norm_aic[i-2]<=0.3: # for rossler clean=0.5
            elif aic_final[i+1,ii]-aic_final[i,ii]>=0 and orinorm_aic[i+1]-orinorm_aic[i]<=Lambda[dim,2]: # for rossler clean=0.5    
                stop = stop+1
            else: 
                break
        
        test_sort = functions['s'+str(stop)]
        InferredResults = pd.concat([InferredResults,test_sort],axis = 1)

    InferredResults = InferredResults.fillna(0)
    print(InferredResults)
    #InferredResults.to_csv('InferredEquation_of_'+str(dim+1)+'-dimension.csv', index=True, header=True)
    return InferredResults, aic_final

def TwoPhaseInference(FuncMatrix, NumDiv, Nnodes, dim, Dim, keep, SampleTime, batchsize, Lambda, plotstart, plotend):
    PhaseOne_series, intercept, imp_cons, imp_no_cons = PhaseOne(FuncMatrix, NumDiv, Nnodes, dim, Dim, keep, Lambda, plotstart, plotend)
    InferredResults, aic_final = PhaseTwo(FuncMatrix, NumDiv, Nnodes, dim, keep, SampleTime, batchsize, Lambda, intercept, imp_cons, imp_no_cons)
    return InferredResults, PhaseOne_series, aic_final, intercept
