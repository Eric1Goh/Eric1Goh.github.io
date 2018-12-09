---
layout: default
permalink: index.html
title: 앙상블 기법 소개
description: "Blogging on ...."
---

 본 포스트는 고려대학교 산업경영공학과 강필성 교수님의 수업을 바탕으로 작성하였습니다.


#####Tree 기반의 대표적인 앙상블(ensemble) 기법인 Random Forest와 Decision Jungle에 대해서 알아보자.

# Random Forest
Random forest는 decision tree 알고리즘을 사용한 bagging 기법의 일종이다.
간단히 말하면, 학습 데이터를 사용하여 여러 개의 Decision tree를 만들고, 만들어진 Decision tree의 결과를 다수결로 최종 결과를 도출하는 알고리즘이다.
![](https://goo.gl/images/724rrJ)

Random forest는 앙상블의 다양성을 위해서 다음과 같은 두가지 방법을 적용하였다.
* Bagging : 각 Decision tree별로 학습 데이터를 다르게 사용
* Randomly chosen predictor variables : Decision tree의 node 분류를 위해 사용할 변수를 randomly 선택하여 모델의 다양성 확보

구현 코드는 다음과 같다.

RandomForest 클래스의 입력값으로 생성할 tree의 개수와 각 tree별로 depth의 최대값을 지정하도록 설계하였다.
각 Decision tree별로 학습할 데이터는 입력 받은 데이터와 동일한 크기로 resampling하였다.
Decision tree에서 사용할 predictor 변수의 개수는 $$$\sqrt{전체 변수 개수}$$$로 고정하였다.

```
class RandomForest:
    def __init__(self, num_tree, max_depth=1):
        self.trees = []
        self.num_tree = num_tree
        self.max_depth = max_depth
    
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
        results = []
        
        for tree in self.trees:
            results.append(tree.predict(test))
        
        return max(set(results), key=results.count)
```


# Decision Tree
![](https://Eric1Goh.github.io/images/training_latent.png)![](https://Eric1Goh.github.io/images/training_test_latent.png)


# Decision Jungle
