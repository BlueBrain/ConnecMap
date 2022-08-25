02/05/2022
configs_4: Using more than 2 components

Split 1:
Step 1: Reversal detector (on component 5)
Step 2: SVM
- C = 0.5
- gamma = 0.05
- thres_size = 5000
- thres gd = 40
- thres ri = 0.05


Split 2:
Step 1: Reversal detector
Step 2: SVM
- C = 0.5
- gamma = 0.05
- thres_size = 300
- thres gd = 40
- thres ri = 0.05


Split 3:
Step 1: cosine distance clustering
- min cluster size = 18
- min samples = 18
(usually 20 but 18 gave a additional region for module1_7)
Step 2: SVM
- C = 0.5
- gamma = 0.05
- thres_size = 100
- thres gd = 40
- thres ri = 0.05

Split 4:
Step 1: cosine distance clustering
- min cluster size = 20
- min samples = 20
Step 2: SVM
- C = 0.5
- gamma = 0.05
- thres_size = 100
- thres gd = 40
- thres ri = 0.05

Split 5:
Step 1: cosine distance clustering
- min cluster size = 20
- min samples = 10
Step 2: SVM
- C = 0.5
- gamma = 0.01
- thres_size = 100
- thres gd = 40
- thres ri = 0.05

Split 6:
Step 1: cosine distance clustering
- min cluster size = 10
- min samples = 10
Step 2: SVM
- C = 0.5
- gamma = 0.05
- thres_size = 100
- thres gd = 40
- thres ri = 0.05