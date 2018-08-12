from sklearn import tree
from sklearn.tree import _tree
# import graphviz
import copy
import random
import logging
import pandas as pd

logging.getLogger().setLevel(logging.INFO)


def regression_tree_tuning(meta_heuristic, dtypes, n_initial_population=50, iter=6, **kwargs):
    """
    Method is based on:
    only suitable for minimization problems!

    "Bartz-Beielstein, Thomas, Konstantinos E. Parsopoulos, and Michael N. Vrahatis.
    "Analysis of particle swarm optimization using computational statistics."
    Proceedings of the International Conference of Numerical Analysis and Applied Mathematics (ICNAAM 2004). 2004."

    Note: No non-integer parameter can be tuned with this algorithm version

    1) Calculate solutions of random instances based of the value ranges
    2) Perform regression tree tuning on a selected meta-heuristic
    3) Calculate new values within the best perform range

    How to use:
    - call this function with the parameters you want to tune.
    - Each parameter should be given specifically with the name
    - Value-Ranges for parameters should be provided in a list
    - Data types must be submitted (currently "int" and "float")
    - The meta-heuristic instance must be initialized with all static parameters
    - The meta-heuristic must have a function "solve"
    - The meta-heuristic must have an attribute "solution"
    - The solution object must have an attribute "value"

    Args:
        meta_heuristic: function of the meta-heuristic you want to tune
        n_initial: size of the initial rule
        dtypes: Data types of the parameters ("int", "float")
        n_initial_population: Size of the initial populatoin
        iter: Number of iterations without changes in the rule-set
        **args: all arguments used

    Returns: tuned final value range
    """

    def get_best_rule_set(tree, rules):
        """
        Get the rule-set of a tree of type sklearn.tree.tree.DecisionTreeRegressor leading to the leaf with the best value

        Args:
            tree: instances of scikit tree
            feature_names: Feature names in the order
            rules: rule set

        """
        # get node with biggest value
        tree_ = tree.tree_

        feature_names = list(rules.keys())
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        # first entry of the array is the lower bound, the second is the upper bound

        def recursive_child_rules_quality(node):
            """
            This function takes the current set of rules used to generate the data.
            It proceeds to go through the tree and set new bounds corresponding to the best leaf.

            IMPORTANT:
            The last rule is most likely the most strikt rule.
            Therefore, if we set one rule once we are not allowed to set it again!

            This is done in an recursive manner by expanding each leaf and selecting the best rule set of the leaf with the
            best return value itself.

            Args:
                node: Child leaf to be selected
                rules: Current state of rules
                type: type of the optimization model (

            Returns: improved rule_set

            """
            nonlocal rules, tree_
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                values_1, rules_1, rule_change_indicator1 = recursive_child_rules_quality(tree_.children_left[node])
                values_2, rules_2, rule_change_indicator2 = recursive_child_rules_quality(tree_.children_right[node])

                # select the best leaf according
                # set the rule set according to the best leaf
                if values_1 < values_2:
                    # we must use deep copy to ensure that both the dictionary as the lists are copied instances
                    values, best_rules, rule_change_indicator = values_1, copy.deepcopy(rules_1), copy.deepcopy(
                        rule_change_indicator1)
                    # check if the rule was already changed if not change and save the change
                    # append new rule (children left means smaller equal -> new upper bound)
                    if not rule_change_indicator[name][1]:
                        best_rules[name][1] = threshold
                        rule_change_indicator[name][1] = True
                else:
                    values, best_rules, rule_change_indicator = values_2, copy.deepcopy(rules_2), copy.deepcopy(
                        rule_change_indicator2)
                    # check if the rule was already changed if not change and save the change
                    # append new rule (children left means bigger -> new lower bound)
                    if not rule_change_indicator[name][0]:
                        best_rules[name][0] = threshold
                        rule_change_indicator[name][0] = True
                return values, best_rules, rule_change_indicator
            else:
                rule_change_indicator = {key: [False, False] for key in rules}
                return tree_.value[node][0][0], rules, rule_change_indicator

        # build each branch and select the rules of the branch with the better objective value
        # return the new rule set!
        return recursive_child_rules_quality(0)[0:2]

    def get_random_instance_values(**kwargs):
        """
        Perform random selection based on the rule set.
        The rule-set layout definition is based on the "regression_tree_tuning" doc_string

        Args:
            **kwargs:

        Returns: random selected values for all kwargs

        """
        nonlocal dtypes

        # build new random instance
        instance_values = {}
        for key in kwargs:
            bounds = kwargs[key]
            dtype = dtypes[key]
            lower_bound = bounds[0]
            upper_bound = bounds[1]

            if type(bounds) is list:
                # select a random float value
                value = random.random() * (upper_bound - lower_bound) + lower_bound
                if dtype == "int":
                    value = int(round(value))
            else:
                raise ValueError("Bounds is neither dict nor tuple object")

            instance_values[key] = value
        return instance_values

    def set_heuristic_attributes(kwargs):
        nonlocal meta_heuristic
        for attribute in kwargs:
            meta_heuristic.__setattr__(attribute, kwargs[attribute])

    def check_stoppage(rule_set, new_rule_set):
        significant_change = False
        # check if at least one siginificatn change can be observed
        for key in rule_set:
            # lower bound must be lower?
            if (new_rule_set[key][0]/rule_set[key][0]) <= 0.95:
                return False
            if (new_rule_set[key][0]/rule_set[key][0]) <= 0.95:
                return False
            if (new_rule_set[key][1]/rule_set[key][1]) <= 0.95:
                return False
            if (rule_set[key][1]/new_rule_set[key][1]) <= 0.95:
                return False
        return True

    heuristic_attribute_values = []

    # 1) INSTANCE GENERATION
    # create new instances and solve them
    for x in range(n_initial_population):
        instance_values = get_random_instance_values(**kwargs)
        heuristic_attribute_values.append(instance_values)

    # 2) EVALUATE ALL INSTANCES
    # data_frame = pd.DataFrame(columns=list(kwargs.keys()) + ["y"])
    data_frame = pd.read_csv("parameter_tuning_results.csv", index_col=0)

    heuristic_solution_values = []
    for x in heuristic_attribute_values:
        set_heuristic_attributes(x)  # call the meta-heuristic with random generated values
        meta_heuristic.solve()
        heuristic_solution_values.append(meta_heuristic.solution.value)

        # save the results
        row = x.copy()
        row.update({"y": meta_heuristic.solution.value})
        data_frame = data_frame.append(row, ignore_index=True)
        data_frame.to_csv("parameter_tuning_results.csv")

    # 3) FIT FIRST REGRESSION TREE
    reg_tree = tree.DecisionTreeRegressor(criterion="mse", splitter="best", max_depth=None,
                                          min_samples_split=2, min_samples_leaf=2)

    x = data_frame.iloc[:,:-1].values
    y = data_frame.loc[:,"y"].values
    reg_tree.fit(x, y)

    # 4) GET FIRST REDUCED RULE SET
    best_leaf_value, reduced_rules = get_best_rule_set(reg_tree, kwargs)

    it = 0  # number of iterations without improvement
    while it < iter:

        # perform new iteration
        new_rule_set = get_random_instance_values(**reduced_rules)
        logging.info(reduced_rules)
        logging.info(new_rule_set)
        set_heuristic_attributes(new_rule_set)
        meta_heuristic.solve()

        # save the results
        row = new_rule_set
        row.update({"y": meta_heuristic.solution.value})
        data_frame = data_frame.append(row, ignore_index=True)
        data_frame.to_csv("parameter_tuning_results_piirim.csv")

        # get the new train data
        x = data_frame.iloc[:, :-1].values
        y = data_frame.loc[:, "y"].values

        # repeat process if necessary
        reg_tree.fit(x, y)
        new_best_leaf_value, new_reduced_rules = get_best_rule_set(reg_tree, reduced_rules)

        # check improvement
        if not check_stoppage(reduced_rules, new_reduced_rules):
            reduced_rules = new_reduced_rules
            it += 0
        else:
            it += 1

        # in each run, save the current values
    return reduced_rules, x, y, reg_tree


if __name__ == "__main__":
    class test_heuristic:
        def __init__(self, a, b):
            self.a = a
            self.b = b
            self.value = 0

        def solve(self):
            self.value = self.a - self.b


    # standard rules
    a = [1, 10]
    b = [1, 10]
    base_rules = {"a": a, "b": b}
    dtypes = {"a": "int", "b": "int"}
    rules, x, y, reg_tree = regression_tree_tuning(test_heuristic, dtypes, a=a, b=b)
    print(rules)

    # print the tree
    #dot_data = tree.export_graphviz(reg_tree, out_file="stuff.pdf")
    #graph = graphviz.Source(dot_data)
    #graph.render("stuff")
