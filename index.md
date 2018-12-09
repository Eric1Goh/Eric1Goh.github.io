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
Decision tree에서 사용할 predictor 변수의 개수는 ![](images/D.png) 로 고정하였다.

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
![](images/entropy.png){: width="80%" height="80%"){: .center}

```Python
class DecisionTree:
    def __init__(self, max_depth=10, log_level=0):
        self.root_node = None      # root node
        self.max_depth = max_depth # tree의 최대 depth
        self.log_level=log_level
        
    def log(self, level, log_data):
        if ( level <= self.log_level):
            print(log_data)
        
    def fit(self, x, y):
        dataset = np.concatenate((x, y), axis=1)
        
        self.root_node = self.build_tree(dataset, self.max_depth)

    def predict(self, test):
        return self.classify(test, self.root_node)

    """
    Randomly selects indexes sqrt(D).
    """
    def random_features(self, nb_features):
        return random.sample(range(nb_features), int(sqrt(nb_features)))

    def divide_dataset(self, dataset, column, value):
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda data: data[column] >= value
        else:
            split_function = lambda data: data[column] == value

        set1 = [data for data in dataset if split_function(data)]
        set2 = [data for data in dataset if not split_function(data)]

        return set1, set2

    def numberOfItems(self, dataset):
        results = {}
        for data in dataset:
            r = data[len(data) - 1]
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results

    def entropy(self, rows):
        results = self.numberOfItems(rows)
        ent = 0.0
        for r in results.keys():
            p = float(results[r]) / len(rows)
            ent = ent - p * np.log2(p)
        return ent

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
# Decision Jungle
