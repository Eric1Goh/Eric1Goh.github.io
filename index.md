---
layout: default
permalink: index.html
title: 앙상블 기법 소개
description: "Blogging on ...."
---

 본 포스트는 고려대학교 산업경영공학과 강필성 교수님의 수업을 바탕으로 작성하였습니다.


#### Tree 기반의 대표적인 앙상블(ensemble) 기법인 Random Forest와 Decision Jungle에 대해서 알아보자.

# Random Forest
Random forest는 decision tree 알고리즘을 사용한 bagging 기법의 일종이다.
간단히 말하면, 학습 데이터를 사용하여 여러 개의 Decision tree를 만들고, 만들어진 Decision tree의 결과를 다수결로 최종 결과를 도출하는 알고리즘이다.

[![](https://eric1goh.github.io/images/random_forest_images.png)](https://goo.gl/images/724rrJ)

Random forest는 앙상블의 다양성을 위해서 다음과 같은 두가지 방법을 적용하였다.
* Bagging : 각 Decision tree별로 학습 데이터를 다르게 사용
* Randomly chosen predictor variables : Decision tree의 node 분류를 위해 사용할 변수를 randomly 선택하여 모델의 다양성 확보


##### 구현 코드는 다음과 같다.
RandomForest 클래스의 입력값으로 생성할 tree의 개수와 각 tree별로 depth의 최대값을 지정하도록 설계하였다.
각 Decision tree별로 학습할 데이터는 입력 받은 데이터와 동일한 크기로 resampling하였다.
Decision tree에서 사용할 predictor 변수의 개수는 ![](https://eric1goh.github.io/images/D.PNG) 로 고정하였다.

```Python
class RandomForest:
    def __init__(self, num_tree, max_depth=1):
        self.trees = []     #학습이 완료된 tree의 list
        self.num_tree = num_tree   #Tree 개수
        self.max_depth = max_depth #Tree의 max depth 
    
    def fit(self, x, y):
        dataset = np.concatenate((x, y), axis=1)
        
        for i in range(self.num_tree):
            tree = DecisionTree(max_depth=self.max_depth)
            
            ## 학습 데이터 resampling
            bagging = resample(dataset, replace=True, n_samples=dataset.shape[0])
            bagging = np.unique(bagging, axis=0)
            
            data_x = bagging[:, 0:dataset.shape[1]-1]
            data_y = bagging[:, dataset.shape[1]-1:]
            
            ## tree 구성
            tree.fit(x=data_x, y=data_y)
            self.trees.append(tree)
    
    def predict(self, test):
        results = []  #Tree별로 예측한 결과를 저장할 list
        
        for tree in self.trees:    #각 tree별로 예측한 결과값을 구함
            results.append(tree.predict(test))
        
        return max(set(results), key=results.count)  #가장 높게 예측한 결과값을 리턴함

    def score(self, x, y):
        results = []
        for row in x:
            results.append(self.predict(row))

        n_true = 0
        for i in range(len(results)):
            if results[i] == test_y[i]:
                n_true += 1

        return (n_true/len(results))
```

다음은 Decision  tree 소스 코드이다.

먼저 tree의 각 node를 구성할 클래스이다. 
 * Leaf node의 경우는 최종 분류값(results)이 저장된다.
 * 그 외 node에서는 분류를 위해 사용한 variable정보(col), 해당 값(value), true인 경우 branch(tb) 그리고 false인 경우의 branch(fb) 정보가 저장된다.

```Python
class Node:
        def __init__(self, col=-1, value=None, results=None, true_branch=None, false_branch=None, depth=-1):
            self.col = col          # 분류에 사용된 variable 정보
            self.value = value      # 분류 경계값
            self.results = results  # 최종 분류값 (for leaf node)
            self.tb = true_branch   # true인 경우 branch
            self.fb = false_branch  # false인 경우 branch
            self.depth = depth
```

Decision tree는 각 영역의 순도가 증가, 불확실성은 감소하는 방향으로 학습을 진행한다. 이때 사용하는 지표는 엔트로피, 지니계수가 있는데, 엔트로피 방식으로 tree를 구현하였다.
엔트로피는 다음과 같이 계산한다.

![](https://eric1goh.github.io/images/entropy.PNG)


```Python
class DecisionTree:
    def __init__(self, max_depth=10, log_level=0):
        self.root_node = None      # root node
        self.max_depth = max_depth # tree의 최대 depth
        self.log_level=log_level
    """ 
     for logging    
    """
    def log(self, level, log_data):
        if ( level <= self.log_level):
            print(log_data)
        
    def fit(self, x, y):
        dataset = np.concatenate((x, y), axis=1)
        
        self.root_node = self.build_tree(dataset, self.max_depth)

    def predict(self, test):
        return self.classify(test, self.root_node)

    """
    split할 변수의 개수는 sqrt(전체 변수 개수)로 고정
    """
    def random_features(self, nb_features):
        return random.sample(range(nb_features), int(sqrt(nb_features)))

    """
    데이터를 지정한 변수(column)와 지정한 값(value)을 기준으로 분리한다.
    값이 숫자인 경우에는 지정한 값고다 크거나 같은 경우와 아닌 경우로 나눈다
    문자열인 경우에는 해당 문자열인 경우와 아닌 경우로 나눈다
    """
    def divide_dataset(self, dataset, column, value):
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda data: data[column] >= value
        else:
            split_function = lambda data: data[column] == value

        set1 = [data for data in dataset if split_function(data)]
        set2 = [data for data in dataset if not split_function(data)]

        return set1, set2

    """
    엔트로피를 계산할 때 필요한 Y 값의 종류와 해당 값이 몇개 인지를 계산한다.
    """
    def numberOfItems(self, dataset):
        results = {}
        for data in dataset:
            r = data[len(data) - 1]
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results

    """
    엔트로피 계산
    """
    def entropy(self, rows):
        results = self.numberOfItems(rows)
        ent = 0.0
        for r in results.keys():
            p = float(results[r]) / len(rows)
            ent = ent - p * np.log2(p)
        return ent

    """
    랜덤하게 선택한 변수를 기준으로 데이터를 value별로 나눌때 엔트로피값이 나누기 전과 비교하여 엔트로피가 가장 감소하는 방향으로 
    tree를 분리한다.
    분리 기준을 저장하는 노드와 최종 분류값을 저장하는 노드(leaf node)로 구성된다.
    """
    def build_tree(self, dataset, depth):
        if len(dataset) == 0:
            return Node()
        if depth == 0:  ## depth constraints
            self.log(1, 'depth={} leaf node(max_depth)'.format(self.max_depth - depth + 1))
            return Node(results=self.numberOfItems(dataset))

        ## 1. choose random feature
        features_indexes = self.random_features(len(dataset[0])-1)
        
        current_score = self.entropy(dataset)
        best_gain = 0.0
        best_condition = None
        best_sub_datasets = None
        
        for col in features_indexes:
            column_values = {}
            for data in dataset:
                column_values[data[col]] = 1
                
            for value in column_values.keys():
                self.log(2, '  {}={}'.format(col, value))
                set1, set2 = self.divide_dataset(dataset, col, value)

                p = float(len(set1)) / len(dataset)
                gain = current_score - p * self.entropy(set1) - (1 - p) * self.entropy(set2)
                self.log(2, '  gain={}'.format(gain))
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_condition = (col, value)
                    best_sub_datasets = (set1, set2)

        if best_gain > 0:
            self.log(1, 'depth={}'.format(self.max_depth - depth + 1))
            self.log(1, '  true branch depth={}'.format(self.max_depth - depth + 1))
            trueBranch = self.build_tree(best_sub_datasets[0], depth - 1)
            self.log(1, '  false branch depth={}'.format(self.max_depth - depth + 1))
            falseBranch = self.build_tree(best_sub_datasets[1], depth - 1)
            return Node(col = best_condition[0],
                        value = best_condition[1],
                        true_branch = trueBranch, 
                        false_branch = falseBranch,
                        depth=(self.max_depth - depth + 1))
        else:
            self.log(1, '  leaf node depth={}'.format(self.max_depth - depth + 1))
            return Node(results=self.numberOfItems(dataset), depth=(self.max_depth - depth + 1))

    """
    tree의 각 노드를 이동하면서 테스트 데이터의 최종 분류값을 찾는다.
    """
    def classify(self, observation, node):
        if node.results is not None:
            return sorted(zip(node.results.values(), node.results.keys()), reverse=True)[0][1]
        else:
            v = observation[node.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= node.value:
                    branch = node.tb
                else:
                    branch = node.fb
            else:
                if v == node.value:
                    branch = node.tb
                else:
                    branch = node.fb
            return self.classify(observation, branch)
```

UCI 공개 데이터를 사용하여 모델의 성능을 평가하였다.
* [Wine Quality Data set ](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
* 총 instance 수 : 4,898
* 독립 변수 : 총 12개 (wine의 산도, 밀도 등)
* 종속 변수 : wine의 품질 지수, 3~9로 구성됨

총 100개의 Decision tree로 구성된 Random forest를 생성하였다.
그리고 테스트 데이터를 사용하여 모델의 성능(accuracy)를 계산하였다.
```Python
forest = RandomForest(max_depth=20, num_tree=100)
forest.fit(x, y)
forest.score(test_x,test_y)
Out : 0.512
```

동일한 데이터셋으로 scikit learn에서 제공하는 Random forest와 성능을 비교하였다.
```Python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=20, criterion="entropy", bootstrap=True)
clf.fit(x, y)

clf.score(test_x, test_y)
-------------------------------------
Out : 0.497
```
성능 차이는 거의 없으나, 모델을 학습할 때 속도 차이가 컸다. Random forest의 tree를 학습할 때 병렬 처리가 필요해 보인다.

# Decision Jungle
Random forest의 경우 뛰어난 성능에도 불구하고 Decision tree의 노드 수가 tree의 깊이 대비 지수에 비례하게 증가하는 단점이 있다. 특정 환경(모바일, 임베디드)에서는 메모리의 제한으로 인해서 tree의 깊이에 제한이 있게 되고 이는 모델의 정확도 저하하는 결과를 가져온다.
Decision Jungle은 node간의 병합을 통해서 node의 개수를 줄여서 Random forest의 한계를 극복하고자 제안된 모델이다.


##### 구현 코드는 다음과 같다.
아래는 고려대학교 산업경영공학과 강성현님이 2015년 R로 작성한 코드를 2016년 동대학원의 김창엽님이 python으로 수정한 코드이다. 해당 코드가 python 2.0을 기반으로 작성되어 있어서 3.0에 맞게 일부 수정하였다.

아래의 코드를 사용하려면 다음과 같은 제약 사항이 있다.
* 종속 변수명을 'Y'로 변경
* 학습 데이터와 테스트 데이터는 pandas DataFrame으로 저장하고 index는 0부터 시작하도록 함

```Python
"""

Created on Sun Nov 27 20:37:53 2016

@original code in R language : SeongHyeon Kang, 2015

@port R to python: Czang Yeob Kim, 2016

"""

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
from scipy.stats import entropy
import time
# for plot_dj
import jgraph


limit_w = 0
limit_d = 0
idx_label = 0
curModel = pd.DataFrame()



#===========================
#         함수 선언부
#===========================

#===========================
# S : d 레벨에서 limit_w 라는 제한이 있을 때 사용할 수 있는 노드 수
#===========================
def S (d, D = None):
    if D == None:
        D = limit_w
        
    return min(pow(2,d), pow(2,D-1))

#===========================
# is_limitWidth :
# d 가 Width의 제한에 걸렸는지 여부를 리턴
# True / False
#===========================
def is_limitWidth (d, D = None):
    # limit_w - User Parameter
    if D == None:
        D = limit_w
    return ( d > (D-1) )
  
#===========================
# cum_numNodes(level) : 
# 지금까지의 레벨까지 사용한 총 노드 수
#===========================
def cum_numNodes (level, w = None):
    num_nodes = 0
    if w == None:
        w = limit_w

    if (level < w):
        for i in range(level):
            num_nodes += pow(2,i)

    else:
        for i in range(w):
            num_nodes += pow(2,i)
        num_nodes = num_nodes + (level-w)*S(level)
        
    return (num_nodes)


#===========================    
# conv_lTon  :
# return nodes # when level is given
# 현재 레벨의 노드 번호 나열
#===========================    
def conv_lTon (level, w = None):
    if w == None:
        w = limit_w

    if (level == 1):
        return 1
    else:
        return range(cum_numNodes(level-1)+1, cum_numNodes(level)+1)

        
#===========================    
# conv_nTol :
# 노드 번호를 이용해서 현재 노드가 속한 레벨을 구한다.
#===========================    
def conv_nTol (node, w = None):
    if w == None:
        w = limit_w
        
    if ( pow(2,w) > node ):
        # 제한에 걸리지 않는 다면
        j = 1
        while (True):
            if(node < pow(2,j)):
                break
            j = j+1
        return (j)
    else:
        return ( ((node-cum_numNodes(w)-1) / S(w)) + w + 1)


#===========================    
# init_edges : Edge 들을 초기화
#===========================  
def init_edges(tree, d = None, w = None):
    if d == None:
        d = limit_d
    if w == None:
        w = limit_w
    
    max_nodes = cum_numNodes(d) # d 레벨까지 사용한 총 노드 수 = max 노드
    for i in range(1,max_nodes+1):
        if( cum_numNodes(w-1) >= i ):
            tree.loc[i-1,'l'] = i*2
            tree.loc[i-1,'r'] = 2*i+1
        else:
            if (i-cum_numNodes(w-1))% S(w) == 0:
                # if i is last nodes of each level
                tree.loc[i-1,'l'] = i+1
                tree.loc[i-1,'r'] = i+S(w) # connect first and last node of next level
            else:
                tree.loc[i-1,'l'] = i+S(w)
                tree.loc[i-1,'r'] = i+S(w)+1  # connect next level(child node)
    return (tree)    


#===========================    
# find_Nc: 현재 노드의 l, r 을 알려준다.
#=========================== 
def find_Nc(i, w = None):
    if w == None:
        w = limit_w
    if( cum_numNodes(w-1) >= i ):
        return [i*2, 2*i+1]
    else:
        if ((i-cum_numNodes(w-1))%S(w, w) ==  0):
            return [i+1, i+S(w, w)]
        else:
            return [i+S(w, w), i+S(w, w)+1]

            

#===========================    
# select_feature : 사용할 feature를 리턴
#=========================== 
def select_feature (f, option = "full", prob = 0):
    # select_feature f는 피처를 나타내는 행 리스트 (y행을 뺀 나머지)
    numX = len(f)
    
    if (option == "full"):
        return (f) # 풀인 경우 그대로 다시 내보냄

    if (option == "sqrt"):
        # 제곱근 만큼 피처를 내보냄
        if (round(math.sqrt(numX), 0) < 2):
            n = 2
        else:
            n = int(round(math.sqrt(numX), 0))           
            return sorted(np.random.choice(f, size=n, replace = False))
        # 복원 추출하지 않고, 제곱근 개의 피처를 리턴
  
    if (option == "prob"):
        # 확률 만큼
        if (round(prob*numX , 0) < 2):
            n = 2
        else:
            n = int(round(prob*numX , 0))
        return (sorted(np.random.choice(f, size=n, replace = False)))

        
    if (option == "log2"):
        # 로그 2 만큼
        if (round(math.log(numX,2), 0) < 2):
            n = 2
        else:
            n = int(round(math.log(numX,2), 0))
        return (sorted(np.random.choice(f, size=n, replace = False)))


#===========================    
# find_majorClass : 범주 중에 더 많은 쪽을 알려줌 (수정)
#===========================     
def find_majorClass(data, idx_label):
    print("[Func] find_majorClass ")

    x = data.iloc[:,idx_label]

    numData = -np.inf
    #print itemfreq(x)  
    for item in itemfreq(x):
        if numData < item[1]:
            numData = item[1]
            majorClass = item[0]
            
    return majorClass     
    #return max(set(x), key=x.count)
   
   
            
#===========================    
# H : 섀넌의 엔트로피 계산
#=========================== 
def H(data, label = None):
    if label == None:
        label = idx_label
    if (len(data) == 0):
        return (0)
    
    return entropy(data["Y"].value_counts().tolist(), qk=None, base=2)


#===========================    
# cal_totalEnt : 현재 레벨 전체의 섀넌의 엔트로피 계산
#=========================== 
def cal_totalEnt(data, idx_label, idx_node):
    ent = 0
    if len(data) == 0:
        return (0)
    #if (is.na(data)) return (0)

    node = np.unique(data.iloc[:, idx_node])

    for i in node:
        subdata = data.loc[ data.iloc[:, idx_node] == i,: ]
        ent = ent + len(subdata) * H(subdata, idx_label[0])

    return (ent)

    
#===========================    
# split_data : 실제 분류하는 부분
#===========================
def split_data (data, idx_feature, idx_label, parent = 'NA'):
    best_theta = best_feature = 0
    min_entropy = np.inf
    
    if H(data, idx_label) == 0:
        return 'NA'
  
    if parent == 'NA':
        print("Decision Tree Algo")
        for i in idx_feature:
            #i = 0
            ## initailize variables 
            #print "[+] Split Feature 가 " + str(i) + "번 째 Feature 일 때"
            data_order = data.iloc[np.argsort(data.iloc[:,i]),:]
            # i 피처에 대한 정렬, order하면 해당 위치에 순서를 알려줌
            # 데이터로 감싸야 원래 형태로 바뀜.
            #data_order는 i 에 대해 정렬된 상태                   
            
            #idx = data_order.loc[:,'idx']
            data_i = data_order.iloc[:, i] # 피처 i 에 대해서 정렬된 데이터
            #print data_i
            #j=0                  
            for j in range(0,len(data)-1):
                # skip if ith and (i+1)th data is same or one of exceptRow j = 1
                if (data_i.iloc[j] == data_i.iloc[j+1]):
                    continue
                
                theta = (data_i.iloc[j] + data_i.iloc[j+1]) / 2      
                # Theta는 중간 값을 취한다. 정렬된 값을 반으로 나눔
                left  = data.loc[ data.iloc[:, idx_feature[i]] <  theta,:]  # Theta보다 작으면 왼쪽
                right = data.loc[ data.iloc[:, idx_feature[i]] >= theta,:]  # Theta보다 크거나 같으면 오른쪽

                # calcurate entropy
                ent_left  = H(left,  idx_label)  # entropy of left nodes
                ent_right = H(right, idx_label)  # entropy of right nodes
                ent_total = (len(left)*ent_left) + (len(right)*ent_right)
                #전체 엔트로피는 왼쪽의 개체수 곱하기 왼쪽 엔트로피 + 오른쪽 개체수 * 오른쪽 엔트로피
                #print ent_total
                # save better parameters 
                if(min_entropy > ent_total ):
                    min_entropy = ent_total
                    best_theta = theta # 엔트로피가 최소가 되는 Theta (전체 하나 위에꺼 까지 전부다 검색)
                    #best_feature = idx_feature[i] # 어떤 피처인지 찾는다.
                    best_feature = i # 어떤 피처인지 찾는다. 바꾼것.


        # result divided dataset
        left  = data.loc[data.iloc[:, best_feature] <  best_theta,:]
        right = data.loc[data.iloc[:, best_feature] >= best_theta,:] 
        
        result = dict({'d' : best_feature, 'theta': best_theta, 'l':left.idx, 'r': right.idx})
        return result
        
    ## decision jungle logic    
    else:
        print("Decision Jungle Algo")
        # extract fixed child nodes(left / right)
        #print "parent : " + str(parent) + "Nc" + str(find_Nc(parent)[0])
        subdata_exRows_l = data.loc[(data.Nc == find_Nc(parent)[0]) & (data.Np != parent),:]
        subdata_exRows_r = data.loc[(data.Nc == find_Nc(parent)[1]) & (data.Np != parent),:]
        
        # extract movable child nodes
        subdata_movable  = data.loc[data.Np == parent,:]
        #print("SUBDATA MOVABLE :" + str(len(subdata_movable)))
        if(len(subdata_movable) == 0):
            best_theta = np.inf
            best_feature = idx_feature[0]

            # result dividing dataset
            left  = subdata_exRows_l
            
            right  = subdata_exRows_r
            
            result = dict({'d' : best_feature, 'theta': best_theta, 'l':left.idx, 'r': subdata_exRows_r})
            return (result)
            
        for i in idx_feature:
            # initailize variables
            #print ("Decision Jungle Feature") + str(i)
            data   = subdata_movable.iloc[ np.argsort(subdata_movable.iloc[:, i]),: ]
            #print "subdata_movable " + str(len(data))
            if(min(data.iloc[:, i]) > 0):
                start = min(data.iloc[:, i])/2
            else:
                start = min(data.iloc[:, i])*2
            if(max(data.iloc[:, i]) > 0):
                end   = min(data.iloc[:, i])*2
            else:
                end   = min(data.iloc[:, i])/2
            
            data_i = []
            data_i.append(start)
            data_i += data.iloc[:, i].tolist()
            data_i.append(end)
            
            for j in range(0, len(data_i)-1):
                
                if data_i[j] == data_i[j+1]:
                    continue
                
                theta = (data_i[j] + data_i[j+1]) / 2
                left  = data.loc[data.iloc[:, i] <  theta,:].append(subdata_exRows_l)
                right  = data.loc[data.iloc[:, i] >=  theta,:].append(subdata_exRows_r)
                
                # calcurate entropy
                
                ent_left  = H(left,  2)  # entropy of left nodes
                ent_right = H(right, 2)  # entropy of right nodes
                ent_total = len(left)*ent_left + len(right)*ent_right
                            
                # save better parameters
                if(min_entropy > ent_total):
                    min_entropy = ent_total
                    best_theta = theta # 엔트로피가 최소가 되는 Theta (전체 하나 위에꺼 까지 전부다 검색)
                    best_feature = i # 어떤 피처인지 찾는다. 수정
        
        # result dividing dataset
        #left  = rbind (subdata_movable.loc[ subdata_movable.iloc[:, best_feature] <  best_theta,: ], subdata_exRows_l)
        left  = subdata_movable.loc[ subdata_movable.iloc[:, best_feature] <  best_theta,: ].append(subdata_exRows_l)
        right = subdata_movable.loc[ subdata_movable.iloc[:, best_feature] >= best_theta,: ]
        
        result = dict({'d' : best_feature, 'theta': best_theta, 'l':left.idx, 'r': right.idx})
        return (result)

        
def plot_dj2(tree, w= None):
    if w == None:
        w = limit_w
   
    depth = conv_nTol(len(tree))
    realDepth = 0

    for i in range(0,depth):
        realDepth += 1
        if sum(tree.loc[conv_lTon(2), "dim"]) <= -2:
            break
    
    width = cum_numNodes(realDepth, w)# + S(w)
    
    m = np.matrix(np.repeat(0, (width) * (width)).reshape((width),(width)))

    node = width-S(w)
    
    for i in range(0,node):
        if tree.dim[i] == -1:
            continue
        if tree.l[i] != -1:
            m[i, tree.l[i]-1] = 1
        if tree.r[i] != -1:
            m[i, tree.r[i]-1] = 1    

    m = m[0:80,0:80]


    g = igraph.Graph.Adjacency((m == 1).tolist())
    
    layout = g.layout_reingold_tilford(mode="all",root=(0,0))
    
    return g, layout 
    
        
def plot_dj(tree, w = None):
    if w == None:
        w = limit_w
    #w = limit_w
    #tree = curModel    
    depth = len(tree)  

    width = depth #+ S(w)

    m = np.matrix(np.repeat(0, (width) * (width)).reshape((width),(width)))

    node = width - S(w)
    
    for i in range(0,node):
        if tree.dim[i] == -1:
            continue
        if tree.l[i] != -1:
            m[i, tree.l[i]-1] = 1
        if tree.r[i] != -1:
            m[i, tree.r[i]-1] = 1
    print(m.shape)
    g = igraph.Graph.Adjacency((m == 1).tolist())
    layout = g.layout_reingold_tilford(mode="all",root=(0,0))
    igraph.plot(g, layout = layout)
    
    return 0


def model_dj (data,               # traning dataset
              idx_feature,        # index of the features 
              idx_label,          # index of the label
              limit_w,            # limit of width (2^w)
              limit_d,            # limit of depth
              op_select = "full", # one of "full, sqrt, prob, log2"
              prob = 0.8):          # if op_select is prob, ratio of choice

    ## dataframe of attributes of a tree
    tree = {'dim'   : np.repeat(-1,cum_numNodes(limit_d, limit_w)),
        'theta' : np.repeat(0,cum_numNodes(limit_d, limit_w)),
        'l'     : np.repeat(0,cum_numNodes(limit_d, limit_w)),
        'r'     : np.repeat(0,cum_numNodes(limit_d, limit_w)),
        'class_': np.repeat(' ',cum_numNodes(limit_d, limit_w))       
        }
    
    tree = pd.DataFrame(tree)

    ## add columns
    
    Np = pd.Series(np.repeat(1, len(data)))
    Nc = pd.Series(np.repeat(1, len(data)))
    
    
    temp = {'Np' : pd.Series(np.repeat(1, len(data))),
            'Nc' : pd.Series(np.repeat(1, len(data))),
            'idx': pd.Series(range(1,len(data)+1))
    }
    
    temp = pd.DataFrame(temp)
    data = pd.concat([data,temp],axis=1)
    
    ## initialize tree edges
    tree = init_edges(tree)
    # 트리는 현재 레벨까지의 총 노드 수 만큼 딤, 쎄타, 엘, 알, 클래스를 초기화하였다.

    #j=1
    ## decision jungle algorithm
    for j in range(1,limit_d+1):
        # j는 1부터 제한 뎁스까지 증가
        print(j)
        
        Np = np.unique((data.loc[:,'Nc']))
      
        print(Np)
        
        #Np = as.numeric(levels( as.factor(data$Nc) )) # return node # of each level
        #Np = conv_lTon(j) # return node # of each level
    
        terminal_flag = True
        #prob = 0.7
        dims = select_feature(f = idx_feature, option = op_select, prob = prob)          
        # feature가 full이면 모든 것 다 리턴
        # idx_feature는 모델을 만들때, 데이터를 읽어 들이고 나서
        # y를 뺀 나머지 인덱스를 선택한다.
        # i = 1
        for i in Np:
            print("Np Node ", str(i))
            subdata = data[data.loc[:,'Nc'] == i]
            idx_subdata = subdata.idx
            tmp = find_majorClass(subdata, idx_label)
            tree.loc[i-1,'class_'] = tmp
            
            if (len(subdata) == 0):
                print("Sub Data 가 0인 경우")
                continue  # 'cause parent node is pure, one of children has all data
            
            
            if( H(subdata, idx_label) == 0):
                # go next loop if entropy is zero
                # %in 연산자
                #print "H 값이 0인 경우"
                data.loc[data.idx.isin(idx_subdata),"Np"] = data.loc[data.idx.isin(idx_subdata),"Nc"]
                data.loc[data.idx.isin(idx_subdata),"Nc"] = find_Nc(i)[0]
                tree.loc[i-1,"dim"] = 1
                tree.loc[i-1,"theta"] = np.inf
                continue
   
            split_info = split_data(subdata, dims, idx_label)

            data.loc[data.idx.isin(split_info['l']), "Np"] = data.loc[data.idx.isin(split_info['l']), "Nc"]
            data.loc[data.idx.isin(split_info['l']), "Nc"] = find_Nc(i)[0]
            data.loc[data.idx.isin(split_info['r']), "Np"] = data.loc[data.idx.isin(split_info['r']), "Nc"]
            data.loc[data.idx.isin(split_info['r']), "Nc"] = find_Nc(i)[1]
                        
            # save split info.
            tree.loc[i-1,"dim"] = split_info['d']
            tree.loc[i-1,"theta"] = split_info['theta']
            terminal_flag = False
    
            # for debug
            print("set threshold : Np is "+ str(i) + " / level is " + str(j))           
            idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
            print(cal_totalEnt(data, idx_label, idx_node = idx_node))
            #print(tree)
    
        print("set threshold : level is " + str(j))
        #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
        
        # terminal condition (if child nodes become pure, go out of loop)
        if terminal_flag == True:
            break 
        
        print(is_limitWidth(j))
        # decision jungle logic below 
        
        if is_limitWidth(j) == False:
            continue
            
        # 2) update best_deminsion & best_theta
    
        for i in Np:
            subdata = data.loc[data.Np == i, :]

            if len(subdata) == 0:
                continue          # if parent node is empty, go to next parent node
            #if (H(subdata, idx_label) == 0):
                #continue # if parent node is pure, we don't need to adjust threshold
                
            split_info = split_data(data, dims, idx_label, i)

            # update Nc and split info.
            data.loc[data.idx.isin(split_info['l']), "Nc"] = find_Nc(i)[0]
            data.loc[data.idx.isin(split_info['r']), "Nc"] = find_Nc(i)[1]
            tree.loc[i-1,"dim"] = split_info['d']
            tree.loc[i-1,"theta"] = split_info['theta']
            # for debug
            #print ("update threshold : Np is " + str(i) + " / level is " str(j))
            #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
            #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
            #print(tree)
            
        print("update threshold : level is " + str(j))
        #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
          

        # 3) update edges
        # left
        for i in Np:
            min_ent = np.inf
            best_edge = 0
            Nc = conv_lTon(j+1)             # index of child nodes in curent level(parent nodes)
            
            if (len(subdata) == 0):
                continue          # if parent node is empty, go to next parent node
                
            #if (H(subdata, idx_label) == 0) next # if parent node is pure, we don't need to adjust threshold
            
            subdata     = data.loc[(data.Np == i) & (data.Nc == find_Nc(i)[0]),:] # extract data of a left edge Np
            idx_subdata = subdata.idx
            
            # update edge
            for k in Nc:
                if (k == find_Nc(i)[1]):
                    continue       # if current child nodes(right)
                data.loc[data.idx.isin(idx_subdata), "Nc"] = k
                idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
                ent = cal_totalEnt(data, idx_label, idx_node = idx_node)
                
                if(min_ent > ent):
                    min_ent = ent
                    best_edge = k
        
            data.loc[data.idx.isin(idx_subdata),"Nc"] = best_edge
            tree.loc[i-1,"l"] = best_edge

            # for debug
            print("update left edge : Np is " + str(i) + " / level is " + str(j))
            #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
            #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
            #print(tree)
            
            
        print("update left edge : level is " +str(j))
        #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
        #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
        
        # right
        for i in Np:
            min_ent = np.inf
            best_edge = 0           
            Nc = conv_lTon(j+1)             # index of child nodes in curent level(parent nodes)
            
            if (len(subdata) == 0):
                continue          # if parent node is empty, go to next parent node
                
            #if (H(subdata, idx_label) == 0) next # if parent node is pure, we don't need to adjust threshold
                        
            subdata     = data.loc[(data.Np == i) & (data.Nc == find_Nc(i)[1]),:] # extract data of a left edge Np
            idx_subdata = subdata.idx

            # update edge
            
            for k in Nc:
                if (tree.loc[i-1,"l"] == k):
                    continue       # if current child nodes(right)
                data.loc[data.idx.isin(idx_subdata), "Nc"] = k
                idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
                ent = cal_totalEnt(data, idx_label, idx_node = idx_node)
                
                if(min_ent > ent):
                    min_ent = ent
                    best_edge = k                    
                    
            data.loc[data.idx.isin(idx_subdata),"Nc"] = best_edge
            if tree.loc[i-1,"l"] > best_edge:
                tree.loc[i-1,"r"] = tree.loc[i-1,"l"]
                tree.loc[i-1,"l"] = best_edge
            else:
                tree.loc[i-1,"r"] = best_edge
                    
            # for debug
            print("update right edge : Np is " + str(i) + " / level is " + str(j))
            #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
            #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
            #print(tree)
            
        print("update right edge : level is " +str(j))
        #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
        #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
        
    return tree
 
    
#===========================    
# predict_dj : 학습한 모델로 실제 predict 하는 부분
#===========================
def predict_dj(tree, data):
    result = list()
    for i in range(0,len(data)):
        x = data.loc[i,:]
        node = 1
        
        while (tree.loc[node-1,"l"] <= len(tree) and tree.loc[node-1,"r"] <= len(tree) ):
            dim_l = tree.loc[(tree.loc[node-1,"l"])-1, "dim"]
            #print dim_l
            dim_r = tree.loc[(tree.loc[node-1,"r"])-1, "dim"]
            #print dim_r
            if (dim_l == -1 & dim_r == -1):   # if next node is empty
                break
            
            if ( x.iloc[tree.dim[node-1]] < tree.theta[node-1]):
                if (tree.loc[(tree.l[node-1])-1, "dim"] == -1):
                    break
                else:
                    node = tree.l[node-1] # go a  left node
            else:
                if (tree.loc[(tree.r[node-1]-1), "dim"] == -1):
                    break
                else:
                    node = tree.r[node-1] # go a right node                
                
        if (tree.loc[node-1, "class_"] == " "):
            print("space err")
        if tree.loc[node-1, "class_"] == "NA":
            print("na err")
        if (tree.loc[node-1, "class_"] == 0):
            print("null err")
        
        #print "Node Num "
        #print node-1
        pred = tree.loc[node-1, "class_"]
        result.append(pred)   
    
    return result
```

Random Forest와 같은 Wine Quality data set으로 모델의 성능을 평가하였다.
* [Wine Quality Data set ](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
* 총 instance 수 : 4,898
* 독립 변수 : 총 12개 (wine의 산도, 밀도 등)
* 종속 변수 : wine의 품질 지수, 3~9로 구성됨

```Python
# 학습 데이터로 모델을 학습함

global idx_label 
global limit_w
global limit_d
global curModel

which = lambda lst:list(np.where(lst)[0])
idx_label = which(data.columns==u"Y")
idx_feature = which(data.columns!=u"Y")

decision_jungle = model_dj(data = data,
                           idx_feature = idx_feature,
                           idx_label   = idx_label,
                           limit_w     = 6,
                           limit_d     = 8)
```

동일한 데이터셋으로 scikit learn에서 제공하는 Random forest와 성능을 비교하였다.
```Python
def score(model, dataset):
    results = predict_dj(model, dataset)
    
    n_true = 0
    for i in range(len(results)):
        if results[i] == dataset['Y'][i]:
            n_true += 1
    return (n_true/len(results))

print(score(curModel, data_test))
-------------------------------------
Out : 0.350
```

모델의 성능은 Random forest보다 낮았다. 

### 결론
