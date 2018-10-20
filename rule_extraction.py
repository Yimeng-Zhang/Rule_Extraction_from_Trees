import sklearn
from sklearn.tree import _tree
from warnings import warn

from rule import Rule

# 2018.10.14 Created by Eamon.Zhang
# 2018.10.15 Add rule filtering function, which allows future functions such as prediction based on selected sets of rules
# 2018.10.15 Warning: DataFrame.Query has a max length limit of 32 to the query string, see https://github.com/PyTables/PyTables/issues/286. 
#            will raise error if rules have >32 conditiions. Unsovlebale yet.

def rule_extract(model, x_test, y_test, sort_key=0, recall_min_c1=0., precision_min_c1=0., recall_min_c0=0., precision_min_c0=0.):
    """
    从 Tree based Algorithm 中提取规则，并在测试集上测试输出准确率/召回率，按条件排序
   
    Parameters
    ----------
    model : 
        用户训练好的模型

    x_test : pandas.DataFrame.
        用来测试的样本的特征集

    y_test : pandas.DataFrame.
        用来测试的样本的y标签
        
    sort_key: 按哪一个指标排序，default = 0
            0： 按label=1样本的 recall 降序
            1： 按label=1样本的 precision 降序
            2： 按label=0样本的 recall 降序
            3： 按label=0样本的 precision 降序    
            
    recall_min_c1：对1类样本的召回率表现筛选
    
    precision_min_c1：对1类样本的准确率表现筛选
    
    recall_min_c0：对0类样本的召回率表现筛选
    
    precision_min_c0：对0类样本的准确率表现筛选

    Returns
    -------
    rules_dict:  list of tuples (rule, recall on 0-class, prec on 0-class, recall on 0-class, prec on 0-class, nb).
        从model里提取的规则（必须是支持的模型）。
        规则可以通过筛选条件来精简。
    
    """
    
    rules_dict = {}
    rules_ = []  
    
    if isinstance(model,(sklearn.tree.tree.DecisionTreeClassifier,sklearn.tree.tree.DecisionTreeRegressor)):
        rules_from_tree = _tree_to_rules(model,x_test.columns)
    
        # 加入规则的评估指标:
        rules_from_tree = [(r, _eval_rule_perf(r, x_test, y_test)) for r in set(rules_from_tree)]
        rules_ = rules_from_tree
        
        # 根据评估指标进行规则筛选
        rules_dict = _rule_filter(rules_,rules_dict, recall_min_c1, precision_min_c1, recall_min_c0, precision_min_c0)
            
        # 按 recall_1 降序排列
        rules_dict = sorted(rules_dict.items(),
                             key=lambda x: (x[1][sort_key]), reverse=True)
        
        return rules_dict
    
    
    elif isinstance(model,(sklearn.ensemble.bagging.BaggingClassifier,
                           sklearn.ensemble.bagging.BaggingRegressor,
                           sklearn.ensemble.forest.RandomForestClassifier,
                           sklearn.ensemble.forest.RandomForestRegressor,
                           sklearn.ensemble.forest.ExtraTreesClassifier,
                           sklearn.ensemble.forest.ExtraTreeRegressor)):
        for estimator in model.estimators_:
            rules_from_tree = _tree_to_rules(estimator,x_test.columns)
            # print(len(rules_from_tree))
            rules_from_tree = [(r, _eval_rule_perf(r, x_test, y_test)) for r in set(rules_from_tree)]
            rules_ += rules_from_tree
            
        
        # 根据评估指标进行规则筛选
        rules_dict = _rule_filter(rules_,rules_dict, recall_min_c1, precision_min_c1, recall_min_c0, precision_min_c0)
            
        # 按 recall_1 降序排列
        rules_dict = sorted(rules_dict.items(),
                             key=lambda x: (x[1][sort_key]), reverse=True)
        
        return rules_dict 
    
    else:
        raise ValueError('Unsupported algorithm')
        return

# 2018.10.15 对规则筛选。目前由于测试集固定，实际上同一条规则的表现不会波动。
def _rule_filter(rules_,rules_dict,recall_min_c1,precision_min_c1,recall_min_c0,precision_min_c0):
    # Factorize rules before semantic tree filtering
    rules_ = [
        tuple(rule)
        for rule in
        [Rule(r, args=args) for r, args in rules_]]
    
    # 根据0/1类样本的recall/precision进行筛选。对于重复出现的规则，则更新其表现。
    for rule, score in rules_:
        if score[0] >= recall_min_c1 and score[1] >= precision_min_c1 and score[2]>=recall_min_c0 and score[3]>=precision_min_c0:
            if rule in rules_dict:
                # update the score to the new mean
                # Moving Average Calculation
                e = rules_dict[rule][4] + 1
                d = rules_dict[rule][3] + 1. / e * (
                    score[3] - rules_[rule][3])
                c = rules_dict[rule][2] + 1. / e * (
                    score[2] - rules_[rule][2])          
                b = rules_dict[rule][1] + 1. / e * (
                    score[1] - rules_[rule][1])
                a = rules_dict[rule][0] + 1. / e * (
                    score[0] - rules_[rule][0])

                rules_dict[rule] = (a, b, c, d, e)
            else:
                rules_dict[rule] = (score[0], score[1], score[2], score[3], 1)
                
    return rules_dict


# 2018.10.14 评估一条单独规则的指标
def _eval_rule_perf(rule, X, y):
    """
    衡量每一条单独规则的评价指标，目前支持 0/1 两类样本的precision/recall
   
    Parameters
    ----------

    rule : str
        从决策树中提取出的单条规则

    X : pandas.DataFrame.
        用来测试的样本的特征集

    y : pandas.DataFrame.
        用来测试的样本的y标签
        
    """
    detected_index = list(X.query(rule).index)
    # print(detected_index)
    if len(detected_index) <= 0:
        warn("rule %s reach no samples" % str(rule))
        return (0.,0.,0.,0.)
    
    y_detected = y[detected_index]
    true_pos = y_detected[y_detected > 0].count()
    false_pos = y_detected[y_detected == 0].count()

    pos = y[y > 0].count()
    neg = y[y == 0].count()
#        recall_0 = str('recall for class 0 is: '+ str(1- (float(false_pos) /neg)))
#        prec_0 = str('prec for class 0 is: ' + str((neg-false_pos) / (len(y)-y_detected.sum())))
#        recall_1 = str('recall for class 1 is: '+ str(float(true_pos) / pos))
#        prec_1 = str('prec for class 1 is: ' + str(y_detected.mean()))
    recall_0 = (1- (float(false_pos) /neg))
    prec_0 = ((neg-false_pos) / (len(y)-y_detected.sum()))
    recall_1 = (float(true_pos) / pos)
    prec_1 = (y_detected.mean())

    return recall_1, prec_1, recall_0, prec_0
    
    
# 2018.10.14 从sklearn 的 tree_ 对象中提取规则
# direct copied from  https://github.com/scikit-learn-contrib/skope-rules/tree/master/skrules
def _tree_to_rules(tree, feature_names):
    """
    Return a list of rules from a tree
    从决策树中提取规则

    Parameters
    ----------
        tree : Decision Tree Classifier/Regressor
        feature_names: list of variable names

    Returns
    -------
    rules : list of rules.
    """
    # XXX todo: check the case where tree is build on subset of features,
    # ie max_features != None

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, base_name):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            symbol = '<='
            symbol2 = '>'
            threshold = tree_.threshold[node]
            text = base_name + ["{} {} {}".format(name, symbol, threshold)]
            recurse(tree_.children_left[node], text)

            text = base_name + ["{} {} {}".format(name, symbol2,
                                                  threshold)]
            recurse(tree_.children_right[node], text)
        else:
            rule = str.join(' and ', base_name)
            rule = (rule if rule != ''
                    else ' == '.join([feature_names[0]] * 2))
            # a rule selecting all is set to "c0==c0"
            rules.append(rule)

    recurse(0, [])

    return rules if len(rules) > 0 else 'True'