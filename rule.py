class Rule:
    """ An object modelizing a logical rule and add factorization methods.
    It is used to simplify rules and deduplicate them.
    将一段决策树规则分解并简化，去除规则中冗余的部分
    eg：
    >>> r = 'feature1 > 3.0 and feature2 < 10.0 and feature3 == 3.5 and feature1 > 4.0'
    >>> result = Rule(r, args=None)
    >>> print(result)
    feature1 > 4.0 and feature2 < 10.0 and feature3 == 3.5

    Parameters
    ----------

    rule : str
        The logical rule that is interpretable by a pandas query.
        一段能被 pandas query 方法解析的规则字符串

    args : object, optional
        Arguments associated to the rule, it is not used for factorization
        but it takes part of the output when the rule is converted to an array.
        额外的参数，不用于规则分解，但会与最后的结果一起输出
    """

    def __init__(self, rule, args=None):
        self.rule = rule
        self.args = args
        self.terms = [t.split(' ') for t in self.rule.split(' and ')]
        self.agg_dict = {}
        self.factorize()
        self.rule = str(self)
  
#    重新定义类中的'=='行为，但没看出哪里派用场了
#    def __eq__(self, other):
#        return self.agg_dict == other.agg_dict

#    暂时不知道用途
#    def __hash__(self):
#        # FIXME : Easier method ?
#        return hash(tuple(sorted(((i, j) for i, j in self.agg_dict.items()))))

    def factorize(self):
        """
        将决策树的规则分解为 字段名 + 判断符号 + 阈值，并进行合并简化
        """
        for feature, symbol, value in self.terms:
            if (feature, symbol) not in self.agg_dict:
                if symbol != '==':
                    self.agg_dict[(feature, symbol)] = str(float(value))
                else:
                    self.agg_dict[(feature, symbol)] = value
            else:
                if symbol[0] == '<':
                    self.agg_dict[(feature, symbol)] = str(min(
                                float(self.agg_dict[(feature, symbol)]),
                                float(value)))
                elif symbol[0] == '>':
                    self.agg_dict[(feature, symbol)] = str(max(
                                float(self.agg_dict[(feature, symbol)]),
                                float(value)))
                else:  # Handle the c0 == c0 case
                    self.agg_dict[(feature, symbol)] = value
        #print(self.agg_dict)

    def __iter__(self):
        yield str(self)
        yield self.args

    def __repr__(self):
        return ' and '.join([' '.join(
                [feature, symbol, str(self.agg_dict[(feature, symbol)])])
                for feature, symbol in sorted(self.agg_dict.keys())
                ])
