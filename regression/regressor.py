import pandas as pd
import numpy as np
import pickle 
import os 
import cv2
import time
import matplotlib.pyplot as plt
from .functions import *
from sklearn.model_selection import train_test_split



class Regressor:
    def __init__(self, c_initial_loc = None):
        self.object = None
        if not c_initial_loc is None:
            self.c = c_initial_loc
        else:
            self.c = np.random.rand(2)
        self.alpha = 0
        self.beta = 0
    def fit(self, Ps, As, lr = 0.1, max_iteration = 10000, print_after= 1, valid_size = 0.25): 
        # print("P: ", Ps)
        # print("A: ", As)
        P_train, P_val, A_train, A_val = train_test_split(Ps, As, test_size = valid_size, random_state=43)
        t_train_start = time.time()
        #print("P_train: \n", P_train)
        #print("A_train: \n", A_train)
        c = self.c
        alpha = self.alpha
        beta = self.beta
        print("____ training ______: ")
        for iter in range(max_iteration):
            alpha = self.calculateParams(c, P_train, A_train)
            # print(alpha)
            g = self.grad(c, alpha, P_train, A_train)
            if np.linalg.norm(g) < 1e-4:
                break
            c -= lr*g
            loss = self.calculateLoss(c, alpha, P_train, A_train)
            if iter % print_after == 0:
                print("iteration {}: ".format(iter))
                print("loss: ", loss)
            #print("__loss-no-bias__: ", loss)
        t_train_end = time.time()
        print("___end training___")
        print('train time: ', (t_train_end - t_train_start)*1000)
        print("Convergence point: ", c)
        print("alpha: ", alpha)
        # cv2.circle(self.drawingBoard, (int(k[0]*10 +300), int(k[1]*10 + 300)), 5, (0,0, 255), -1)
        # cv2.imshow('abc', self.drawingBoard)
        # cv2.waitKey()
        # cv2.destroyAllWindows()  
        err_xs = []
        err_ys = []
        errs = []
        print("______validating_______")
        t_valid_start = time.time()
        for i, p in enumerate(P_val):
            a = A_val[i]
            reg_a = p + (c-p)/alpha
            print('original: ', p)
            print('regression result: ', reg_a)
            print('original error: ', np.linalg.norm(p-a))
            print('after regress error:', np.linalg.norm(reg_a -a))
            print('ground truth: ', a)
            err_x = np.abs((reg_a - a)[0])
            err_y = np.abs((reg_a - a)[1]) 
            err = np.linalg.norm(reg_a - a)
            errs.append(err)
            err_xs.append(err_x)
            err_ys.append(err_y)
        t_valid_end = time.time()
        mean_err = np.mean(errs)
        print("______end validating_______")
        print('mean err: {}'.format(mean_err))
        print("valid time: ", (t_valid_end - t_valid_start)*1000)
        print("valid time avg: ", (t_valid_end - t_valid_start)*200)
        dis = []
        err = []
        for i, p in enumerate(Ps):        
            a = As[i]
            dis.append(np.linalg.norm(c - a)*10)
            err.append(np.linalg.norm(p - a)*10)
        
        self.c = c
        self.alpha = alpha
        self.beta = beta
        # params_path = "{}{}_fold{}_mve{}.pkl".format(self.out_dir, time.ctime(time.time()), j+1, np.round(mean_err,2))
        #last_params_path = "{}last.pkl".format(self.out_dir)
        # pickle.dump(params, open(params_path, 'wb'))
        #pickle.dump(params, open(last_params_path, 'wb'))
    
    def save(self, path):
        params = {
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta
        }
        pickle.dump(params, open(path, "wb"))
    def load(self, path):
        params = pickle.load(open(path, "rb"))
        self.c = params['c']
        self.alpha = params['alpha']
        self.beta = params['beta']
        return self

    def predict(self, p):
        # print(self.c)
        # print(self.alpha)
        a = p + (self.c-p)/self.alpha
        return a

    def grad(self, k, q, X, Y):
        dJ = 0
        for i, x in enumerate(X):
            y = Y[i]
            #print(k-x)
            #print('dJ: ', dJ)
            dJ += (k-x)*4*(np.linalg.norm(k-x)/np.linalg.norm(x-y) -q)/(np.linalg.norm(x-y)*np.linalg.norm(k-x))
        dJ = dJ/len(X)
        #print('dJ: ', dJ)
        return dJ
    def calculateParams(self, c, X, Y):
        # print(X.shape)
        # print(Y.shape)
        alpha_sum = 0
        #print(Y)
        for i, x in enumerate(X):
            y = Y[i]
            # print(x, y)
            # print(x.shape)
            # print(y.shape)
            # print(c.shape)
            alpha_sum += np.linalg.norm(c-x)/np.linalg.norm((x-y))
        # print(alpha_sum)
        alpha =  alpha_sum/len(X)
        return alpha 
    def calculateLoss(self, k, q, X, Y):
        loss = 0
        for i, x in enumerate(X):
            y = Y[i]
            qi = np.linalg.norm((k-x))/np.linalg.norm((x-y))
            loss += 2*(q - qi)**2/len(X)
        return loss


class MultiRegressor:
    def __init__(self, args):
        self.out_dir = args.out_dir
        self.params_dir = args.out_dir + 'parameters/regression/'
        self.regressors = dict()
        if args.regression_mode == "train":
            pass
        elif args.regression_mode == "test":
            self.load(self.params_dir, args.classes)
        pass
    def fit(self, Ps, As, c_initial_loc):
        assert len(Ps.keys()) == len(As.keys())
        for key in Ps.keys():
            if not np.isnan(np.sum(As[key])):
                path = "{}{}/{}.pkl".format(self.params_dir, key, time.ctime(time.time()))
                last_path = "{}{}/last.pkl".format(self.params_dir, key)
                self.regressors[key] = Regressor(c_initial_loc)
                self.regressors[key].fit(Ps[key], As[key])     

        self.save()
    def predict(self, class_name, loc):
        if not class_name in self.regressors.keys():  
            return None
        reg = self.regressors[class_name]
        return reg.predict(loc)
    
    def save(self):
        params_dir = self.params_dir
        for k in self.regressors.keys():
            k_params_dir = os.path.join(params_dir, k)
            if not os.path.exists(k_params_dir):
                os.makedirs(k_params_dir)
            params_path = "{}/{}.pkl".format(k_params_dir, time.ctime(time.time()))
            last_params_path = "{}/last.pkl".format(k_params_dir)
            self.regressors[k].save(params_path)
            self.regressors[k].save(last_params_path)

    def load(self, path, classes):
        for c in classes:
            c_path = "{}{}/last.pkl".format(path, c) 
            if os.path.exists(c_path):
                self.regressors[c] = Regressor().load(c_path)

