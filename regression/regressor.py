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
    def __init__(self, args):
        self.out_dir = args.out_dir
        self.params_path = args.out_dir + 'parameters/regressor/'
    
    def fit(self, P, A, lr = 0.1, max_iteration = 1000, print_after= 20, num_folds = 4): 
         
        folds = foldSplit(P, A, num_folds)
        for j in range(num_folds):
            print("_________Fold {}: __________".format(j+1))
            P_train, P_val, A_train, A_val = ithFoldTrainTestSplit(folds, j)
            
            c = (np.random.rand(2) - 0.5)*30
            loss = 0
            alpha = 0
            t_train_start = time.time()
            print("____ training ______: ")
            for iter in range(max_iteration):
                alpha = self.calculateParams(c, P_train, A_train)
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
                print('regression result: ', reg_a)
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
            for i, p in enumerate(P):        
                a = A[i]
                dis.append(np.linalg.norm(c - a)*10)
                err.append(np.linalg.norm(p - a)*10)
            params = {
                'c': c,
                'alpha': alpha
            }
            # params_path = "{}{}_fold{}_mve{}.pkl".format(self.out_dir, time.ctime(time.time()), j+1, np.round(mean_err,2))
            last_params_path = "{}last.pkl".format(self.out_dir)
            # pickle.dump(params, open(params_path, 'wb'))
            #pickle.dump(params, open(last_params_path, 'wb'))

    
    def predict(self, p):
        params = pickle.load(open(self.last_params, "rb"))
        c = params['c']
        alpha = params['alpha']
        a = p + (c-p)/alpha
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
    def calculateParams(self, k, X, Y):
        q_sum = 0
        for i, x in enumerate(X):
            y = Y[i]
            q_sum += np.linalg.norm((k-x))/np.linalg.norm((x-y))
        q =  q_sum/len(X)
        return q 
    def calculateLoss(self, k, q, X, Y):
        loss = 0
        for i, x in enumerate(X):
            y = Y[i]
            qi = np.linalg.norm((k-x))/np.linalg.norm((x-y))
            loss += 2*(q - qi)**2/len(X)
        return loss
