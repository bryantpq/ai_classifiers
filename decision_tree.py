'''
Bryan Quah, Matthew Xu
CSE 415 Project
Decision Tree implementation and associated functions
Reference: https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
'''

class Question:
    def __init__(self, col, value):
        '''
        A question used to determine if a given column in the data matches the given value.
        '''
        self.col = col
        self.value = value

    def ask(self, row):
        '''
        Returns true if the column in the given row satisfies this question.
        '''
        data = row[self.col]
        return data >= self.value if self.is_numeric(data) else data == self.value

    def is_numeric(self, val):
        '''
        Returns true if a value is numeric, i.e. int or float
        '''
        return isinstance(val, int) or isinstance(val, float)


def split_data(data, question):
    '''
    Checks each row in data if it satisfies the given question. Returns rows that satisfies
    the question and rows that don't satisfy the question.
    '''
    pass_rows = []
    fail_rows = []
    for row in data:
        if question.ask(row):
            pass_rows.append(row)
        else:
            fail_rows.append(row)
    return pass_rows, fail_rows


class QuestionNode:
    def __init__(self, question, pass_branch, fail_branch):
        '''
        Internal node of the decision tree, which keeps track of what question to ask,
        the rows that pass the question and the rows that fail the question.
        '''
        self.question = question
        self.pass_branch = pass_branch
        self.false_branch = false_branch


class AnswerNode:
    def __init__(self, rows, col=-1):
        '''
        Leaf node of the decision tree, which keeps count of how many of each label
        is present in the rows.
        '''
        self.counts = self.count_labels(rows, col)


def count_labels(data, col=-1):
    '''
    Returns a dictionary of the counts of labels in each row in the data.
    '''
    res = {}
    for row in data:
        label = row[col]
        if label not in res:
            res[label] = 0
        res[label] += 1
    return res


def compute_gini(data, col=-1):
    '''
    Returns the gini impurity for the given data
    '''
    labels = count_labels(data, col)
    impurity = 1
    for label in labels:
        label_probability = labels[label] / float(len(data))
        impurity -= label_probability**2
    return impurity


def information_gain(left, right, cur_impurity):
    '''
    The impurity of the initial data minus the weighted impurity of two
    candidate child data sets.
    '''
    p = float(len(left)) / (len(left) + len(right))
    return cur_impurity - p * compute_gini(left) - (1 - p) * compute_gini(right)


def find_best_split(data):
    '''
    Finds the best question to ask for the current data set by calculating the information
    gained from splitting on every possible attribute and value.
    '''
    best_gain = 0
    best_ques = None
    cur_impurity = compute_gini(data)
    num_attributes = len(data[0]) - 1

    for col in range(num_attributes):
        # finds the number of possible values for an attribute/column
        values = set([row[col] for row in data]) 
        for value in values:
            cur_ques = Question(col, value)
            pass_rows, fail_rows = split_data(data, cur_ques)
            if len(pass_rows) == 0 or len(false_rows) == 0: continue 

            cur_gain = information_gain(pass_rows, fail_rows, cur_impurity)

            if cur_gain > best_gain:
                best_gain = cur_gain
                best_ques = cur_ques

    return best_gain, best_ques


def build_decision_tree(data):
    '''
    Recursive algorithm to build the decision tree for the given data.
    '''
    gain, ques = find_best_split(data)

    if gain == 0: return AnswerNode(data)

    # There is still a better question to split the data
    pass_rows, fail_rows = split_data(data)
    pass_rows = build_decision_tree(pass_rows)
    fail_rows = build_decision_tree(fail_rows)

    return QuestionNode(ques, pass_rows, fail_rows)


def classify(root, data, label=True):
    '''
    Uses the given decision tree to classify a single row of data.
    Returns a dictionary of the results.
    '''
    if isinstance(root, AnswerNode):
        if not label:
            return root.counts
        else:
            max_label = None
            max_val = 0
            for k in root.counts.keys():
                if root.counts[k] > max_val:
                    max_label = k
            return max_label

    if root.question.ask(data):
        classify(root.true_branch, data)
    else:
        classify(root.false_branch, data)
