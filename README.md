# Rule_Extraction_From_Trees



基于 Sklearn 的官方项目 skope-rules 改写 ( [Skope-rules 地址](https://github.com/scikit-learn-contrib/skope-rules) )，旨在从 Tree-based 算法中提取明晰的可解释的规则语句，目前只支持二分类任务。

与 skope-rules 相比，具体改动如下：
- 将模型训练与规则提取部分分离，本项目只负责从 Tree 中提取规则
- 允许在单独的测试集上测试每一条规则的性能
- 输出 class 0/1 两类样本的 Recall 和 Precision
- 支持算法包括：单颗决策树, BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreeRegressor





Dependencies
------------

skope-rules requires:

- Python (>= 2.7 or >= 3.3)
- NumPy (>= 1.10.4)
- SciPy (>= 0.17.0)
- Pandas (>= 0.18.1)
- Scikit-Learn (>= 0.17.1)

