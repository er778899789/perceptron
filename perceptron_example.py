#! /usr/bin/env python  
#coding=utf-8  
import numpy as np  
#感知器分類的學習  
class Perceptron:  
    ''''' 
    eta:學習速率
    n_iter:權重向量的訓練次數 
    w_:權重向量 
    errors_:記錄判断出錯的次數
     
    '''  
    def __init__(self,eta=0.01,n_iter=10):  
        self.eta=eta  
        self.n_iter=n_iter  
          
    def fit(self,X,y):  
        ''''' 
        輸入訓練數據X，訓練神經元，X輸入樣本，y樣本分類 
        x=[[1,2],[4,5]] 
        y=[-1,1] 
        '''  
        #初始化權重向量,加1是因為W0
        #X.shape為陣列之形狀
        self.w_=np.zeros(1+X.shape[1])  
        #print(self.w_)#w_=[0,0,0]  
        self.errors_=[]  
          
        for i in range(self.n_iter):  
            errors=0  
            ''''' 
            zip(X,y)=[[1,2,-1],[4,5,1]] 
            '''  
            for xi,target in zip(X,y):#每次遞迴使用一个樣本去更新W  
                #相當於update=$*(y-y'),這裡使用預測的結果進行誤差判斷  
                update=self.eta*(target-self.predict(xi))  
                ''''' 
                xi是一个向量[1,2] 
                update是一個數字 
                update*xi等於
                w1'=x1*update;w2'=x2*update 
                '''  
                self.w_[1:]+=update*xi  
                self.w_[0]+=update*1  
                #印出更新的W_  
                #print self.w_  
                #統計 判斷正確與否的次數  
                errors+=int(update!=0)  
                self.errors_.append(errors)  
                  
    def net_input(self,X):  
        ''''' 
        z=w0*1+w1*x1+w2x2+...+wm*xm 
        其中x0=1（一般w0=0,x0=1） 
        '''  
        return np.dot(X,self.w_[1:])+self.w_[0]*1  
      
    def predict(self,X):#相當於sign()函数  
        ''''' 
        y>=0--->1 
        y<0---->-1 
        '''  
        return np.where(self.net_input(X)>=0.0,1,-1)  
