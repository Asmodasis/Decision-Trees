import math
import numpy as np


class BinaryTree:
    """ If left == None and right == None then we have a leaf => data is an int (0 or 1) and represents the outcome.
    If at least one of left or right is another BinaryTree then we have a node => data is an int starting at 0
    indicating the feature (index) on which is split. split_position is only used for trees with real values. """
    data = None
    split_position = None
    left = None
    right = None

    def __init__(self, data, left, right, split_position=None):
        self.data = data
        self.left = left
        self.right = right
        self.split_position = split_position

    def print(self):
        print(str(self) + " *** BinaryTree:  data: " + str(self.data) + ", split_position: " + str(
                self.split_position) + ", left: " + str(self.left) + "***" + str(self.left.print() if self.left is not None else self.left) + ", right: " + str(self.right) + "***" + str(self.right.print() if self.right is not None else self.right))


# takes a list and an index, Removes the element at that index on said list
def ourRemove(L, indexToRemove):
    replaceList = []                        # create a replacement list

    for i in range(len(L)):                 # loop over L
        if i is not indexToRemove:
            replaceList.append(L[i])        # add all elements besides the one at IndexToRemove
    return replaceList


def calculate_entropy(p1, p2):
    # We only use binary trees. Therefore we only need two probabilities here.
    return (-p1 * math.log2(p1) if p1 != 0 else 0) - (p2 * math.log2(p2) if p2 != 0 else 0)


def calculate_information_gain(h, h1, h2, p1, p2):
    # We only use binary trees. Therefore we only need two probabilities / entropies here.
    return h - p1 * h1 - p2 * h2


def get_number_of_x_y_labels(Y):
    number_of_zero_labels = 0
    number_of_one_labels = 0
    for row in Y:
        if row[0] == 0:
            number_of_zero_labels += 1
        else:
            number_of_one_labels += 1
    return number_of_zero_labels, number_of_one_labels


def get_samples_and_labels_with_feature_value(X, Y, feature_value, feature_index):
    """ We want only the samples with value 0 xor 1. """
    sample_subset = []
    label_subset = []

    if Y is not None:
        for i in range(len(Y)):
            if X[i][feature_index] == feature_value:
                sample_subset.append(X[i])
                label_subset.append(Y[i])

    return sample_subset, label_subset


def DT_train_binary2(X, Y, max_depth, remaining_features):
    """ remaining_features is a list indicating which features we haven't used yet. Feature index starts at 0. """
    # End of recursion / Stop criterion
    if max_depth == 0 or len(Y) == 0 or remaining_features is None or len(remaining_features) == 0:
        n_zero_labels, n_one_labels = get_number_of_x_y_labels(Y)
        return BinaryTree(0 if n_zero_labels >= n_one_labels else 1, None, None)  # leaf

    else:
        # Check if all labels are the same. If so then return a tree containing only one leaf.
        are_all_labels_the_same = True
        temp_label_value = None
        for row in Y:
            if temp_label_value is not None:
                if row[0] != temp_label_value:
                    are_all_labels_the_same = False
                    break
            else:
                temp_label_value = row[0]

        if are_all_labels_the_same:
            return BinaryTree(temp_label_value, None, None)  # leaf

        # Greedy algorithm
        # Calculate the entropy of the entire dataset or the remaining dataset
        n_zero_labels, n_one_labels = get_number_of_x_y_labels(Y)
        p1 = n_zero_labels / len(Y)
        p2 = n_one_labels / len(Y)

        h_ = calculate_entropy(p1, p2)

        # Split the training data using each feature
        best_ig = 0
        best_feature_to_split_on = None
        remaining_left_samples_for_best_ig = []
        remaining_right_samples_for_best_ig = []
        remaining_left_labels_for_best_ig = []
        remaining_right_labels_for_best_ig = []
        for feature_index in range(len(X[0])):
            # only use remaining features here
            if feature_index not in remaining_features:
                continue

            # calculate h1 and h2
            sample_subset1, label_subset1 = get_samples_and_labels_with_feature_value(X, Y, 0, feature_index)  # 0 / NO / left
            sample_subset2, label_subset2 = get_samples_and_labels_with_feature_value(X, Y, 1, feature_index)  # 1 / YES / right

            n_zero_labels1, n_one_labels1 = get_number_of_x_y_labels(label_subset1)
            n_zero_labels2, n_one_labels2 = get_number_of_x_y_labels(label_subset2)

            p11 = (n_zero_labels1 / (n_zero_labels1 + n_one_labels1)) if (n_zero_labels1 + n_one_labels1) != 0 else 0
            p12 = (n_one_labels1 / (n_zero_labels1 + n_one_labels1)) if (n_zero_labels1 + n_one_labels1) != 0 else 0

            p21 = (n_zero_labels2 / (n_zero_labels2 + n_one_labels2)) if (n_zero_labels2 + n_one_labels2) != 0 else 0
            p22 = (n_one_labels2 / (n_zero_labels2 + n_one_labels2)) if (n_zero_labels2 + n_one_labels2) != 0 else 0

            h1 = calculate_entropy(p11, p12)
            h2 = calculate_entropy(p21, p22)

            # calculate IG
            n_rows_0 = len(label_subset1)
            n_rows_1 = len(label_subset2)

            p1_ig = n_rows_0 / len(Y)
            p2_ig = n_rows_1 / len(Y)

            ig = calculate_information_gain(h_, h1, h2, p1_ig, p2_ig)

            # store best IG, the subsets (samples and labels) and the feature_index
            if ig >= best_ig:
                best_ig = ig
                best_feature_to_split_on = feature_index
                remaining_left_samples_for_best_ig = sample_subset1
                remaining_right_samples_for_best_ig = sample_subset2
                remaining_left_labels_for_best_ig = label_subset1
                remaining_right_labels_for_best_ig = label_subset2

        if best_ig == 0:
            # If best_ig == 0 then we have to return a leaf here because we didn't achieve anything.
            n_zero_labels, n_one_labels = get_number_of_x_y_labels(Y)
            return BinaryTree(0 if n_zero_labels >= n_one_labels else 1, None, None)  # leaf
        else:
            # Choose to split on the feature that gives the maximum IG
            left = DT_train_binary2(remaining_left_samples_for_best_ig, remaining_left_labels_for_best_ig, max_depth-1 if max_depth != -1 else max_depth, ourRemove(remaining_features, best_feature_to_split_on))
            right = DT_train_binary2(remaining_right_samples_for_best_ig, remaining_right_labels_for_best_ig, max_depth-1 if max_depth != -1 else max_depth, ourRemove(remaining_features, best_feature_to_split_on))
            return BinaryTree(best_feature_to_split_on, left, right)  # node


def DT_train_binary(X, Y, max_depth):
    remaining_features = []

    if X is None or len(X) == 0:
        print("There are no samples, therefore we can not build a decision tree.")
        return
    elif Y is None or len(Y) == 0:
        print("There are no labels, therefore we can not build a decision tree.")
        return
    else:
        for i in range(len(X[0])):
            remaining_features.append(i)
        return DT_train_binary2(X.tolist(), Y.tolist(), max_depth, remaining_features)


def traverse_tree(sample, DT):
    # There is no case where DT.left xor DT.right can be None and BinaryTree being a leaf
    if DT.left is None and DT.right is None:
        return DT.data  # data is the prediction
    else:
        if sample[DT.data] == 0:  # data is the feature index
            return traverse_tree(sample, DT.left)
        else:
            return traverse_tree(sample, DT.right)


def DT_test_binary(X, Y, DT):
    predictions = []

    if X is None or len(X) == 0:
        print("There are no samples, therefore we can not test the decision tree.")
        return
    elif Y is None or len(Y) == 0:
        print("There are no labels, therefore we can not test the decision tree.")
        return
    else:
        for sample in X:
            predictions.append(traverse_tree(sample, DT))

        # calculate accuracy
        n_rows, n_cols = Y.shape
        n_correct = 0
        for i in range(n_rows):
            if predictions[i] == Y[i, 0]:
                n_correct += 1

        return n_correct / n_rows


def DT_make_prediction(x, DT):
    return traverse_tree(x, DT)


""" real data"""


def compute_IG_for_split_position(h_, len_Y, n_zero_lessequal, n_one_lessequal, n_zero_greater, n_one_greater):
    # calculate h1 and h2
    # left
    p11 = (n_zero_lessequal / (n_zero_lessequal + n_one_lessequal)) if (n_zero_lessequal + n_one_lessequal) != 0 else 0
    p12 = (n_one_lessequal / (n_zero_lessequal + n_one_lessequal)) if (n_zero_lessequal + n_one_lessequal) != 0 else 0

    # right
    p21 = (n_zero_greater / (n_zero_greater + n_one_greater)) if (n_zero_greater + n_one_greater) != 0 else 0
    p22 = (n_one_greater / (n_zero_greater + n_one_greater)) if (n_zero_greater + n_one_greater) != 0 else 0

    h1 = calculate_entropy(p11, p12)
    h2 = calculate_entropy(p21, p22)

    # calculate IG
    n_rows_left = n_zero_lessequal + n_one_lessequal
    n_rows_right = n_zero_greater + n_one_greater

    p1_ig = n_rows_left / len_Y
    p2_ig = n_rows_right / len_Y

    return calculate_information_gain(h_, h1, h2, p1_ig, p2_ig)


def get_samples_and_labels_at_split_position(X, Y, split_position, feature_index):
    sample_subset_left = []
    label_subset_left = []

    sample_subset_right = []
    label_subset_right = []

    if Y is not None:
        for i in range(len(Y)):
            if X[i][feature_index] <= split_position:
                sample_subset_left.append(X[i])
                label_subset_left.append(Y[i])
            else:
                sample_subset_right.append(X[i])
                label_subset_right.append(Y[i])

    return sample_subset_left, label_subset_left, sample_subset_right, label_subset_right


def DT_train_real2(X, Y, max_depth, remaining_features):
    """ remaining_features is a list indicating which features we haven't used yet. Feature index starts at 0. """
    # End of recursion / Stop criterion
    if max_depth == 0 or len(Y) == 0 or remaining_features is None or len(remaining_features) == 0:
        n_zero_labels, n_one_labels = get_number_of_x_y_labels(Y)
        return BinaryTree(0 if n_zero_labels >= n_one_labels else 1, None, None)  # leaf

    else:
        # Check if all labels are the same. If so then return a tree containing only one leaf.
        are_all_labels_the_same = True
        temp_label_value = None
        for row in Y:
            if temp_label_value is not None:
                if row[0] != temp_label_value:
                    are_all_labels_the_same = False
                    break
            else:
                temp_label_value = row[0]

        if are_all_labels_the_same:
            return BinaryTree(temp_label_value, None, None)  # leaf

        # Greedy algorithm
        # Calculate the entropy of the entire dataset or the remaining dataset
        n_zero_labels, n_one_labels = get_number_of_x_y_labels(Y)
        p1 = n_zero_labels / len(Y)
        p2 = n_one_labels / len(Y)

        h_ = calculate_entropy(p1, p2)

        # Split the training data using each feature
        best_ig = 0
        best_feature_to_split_on = None
        best_split_position_for_best_feature = None

        for feature_index in range(len(X[0])):
            # only use remaining features here
            if feature_index not in remaining_features:
                continue

            # I. Compute the decision boundary with the best IG
            # 1. Sort the samples (and labels) by feature values ascending
            sorted_X, sorted_Y = zip(*sorted(zip(X, Y), key=lambda l: l[0][feature_index]))

            # 2. Linearly scan these feature values + update the counts of 0s and 1s for NO and YES each + compute IG
            split_step = 0.1
            split_start_position = X[0][feature_index] - split_step
            split_end_position = X[len(X) - 1][feature_index]
            current_split_position = split_start_position

            n_zero_lessequal = 0
            n_one_lessequal = 0
            n_zero_greater, n_one_greater = get_number_of_x_y_labels(sorted_Y)

            best_ig_for_split = 0
            best_split_position = None

            sample_index = 0

            # This loop has linear complexity O(n1 + n2) with n1 = number of split positions and n2 = number of samples
            while current_split_position <= split_end_position:
                while sorted_X[sample_index][feature_index] <= current_split_position:
                    # update counts for 0s and 1s
                    if sorted_X[sample_index][feature_index] <= current_split_position:
                        if sorted_Y[sample_index][0] == 0:
                            n_zero_lessequal += 1
                            n_zero_greater -= 1
                        else:
                            n_one_lessequal += 1
                            n_one_greater -= 1
                    # We don't need the else case here because the lists are sorted so there won't be a transition from
                    # "less than or equal to" to "greater than".

                    # Go to the next sample until the current_split_position is reached.
                    sample_index += 1

                # Compute the IG for the current split position
                ig_for_split = compute_IG_for_split_position(h_, len(sorted_Y), n_zero_lessequal, n_one_lessequal,
                                                                 n_zero_greater, n_one_greater)

                # 3. Choose the split value/position with the highest IG
                # Only save if the IG for the current split position is better than or equal to the IG for the previous
                # best split position
                if ig_for_split >= best_ig_for_split:
                    best_ig_for_split = ig_for_split
                    best_split_position = current_split_position

                # Go to the next split position.
                current_split_position += split_step

            # II. Choose the feature with the highest IG
            # store best IG, the subsets (samples and labels) and the feature_index
            if best_ig_for_split >= best_ig:
                best_ig = best_ig_for_split
                best_feature_to_split_on = feature_index
                best_split_position_for_best_feature = best_split_position

        if best_ig == 0:
            # If best_ig == 0 then we have to return a leaf here because we didn't achieve anything.
            n_zero_labels, n_one_labels = get_number_of_x_y_labels(Y)
            return BinaryTree(0 if n_zero_labels >= n_one_labels else 1, None, None)  # leaf
        else:
            # Compute the feature and label subsets for the left and right side of the tree.
            remaining_left_samples_for_best_ig, remaining_left_labels_for_best_ig, remaining_right_samples_for_best_ig, remaining_right_labels_for_best_ig = get_samples_and_labels_at_split_position(X, Y, best_split_position_for_best_feature, best_feature_to_split_on)

            # Choose to split on the feature that gives the maximum IG
            left = DT_train_real2(remaining_left_samples_for_best_ig, remaining_left_labels_for_best_ig,
                                    max_depth - 1 if max_depth != -1 else max_depth,
                                    ourRemove(remaining_features, best_feature_to_split_on))
            right = DT_train_real2(remaining_right_samples_for_best_ig, remaining_right_labels_for_best_ig,
                                     max_depth - 1 if max_depth != -1 else max_depth,
                                     ourRemove(remaining_features, best_feature_to_split_on))
            return BinaryTree(best_feature_to_split_on, left, right, best_split_position_for_best_feature)  # node


def DT_train_real(X, Y, max_depth):
    remaining_features = []

    if X is None or len(X) == 0:
        print("There are no samples, therefore we can not build a decision tree.")
        return
    elif Y is None or len(Y) == 0:
        print("There are no labels, therefore we can not build a decision tree.")
        return
    else:
        for i in range(len(X[0])):
            remaining_features.append(i)
        return DT_train_real2(X.tolist(), Y.tolist(), max_depth, remaining_features)


def traverse_tree_real(sample, DT):
    # There is no case where DT.left xor DT.right can be None and BinaryTree being a leaf
    if DT.left is None and DT.right is None:
        return DT.data  # data is the prediction
    else:
        if sample[DT.data] <= DT.split_position:  # data is the feature index
            return traverse_tree_real(sample, DT.left)
        else:
            return traverse_tree_real(sample, DT.right)


def DT_test_real(X, Y, DT):
    predictions = []

    if X is None or len(X) == 0:
        print("There are no samples, therefore we can not test the decision tree.")
        return
    elif Y is None or len(Y) == 0:
        print("There are no labels, therefore we can not test the decision tree.")
        return
    else:
        for sample in X:
            predictions.append(traverse_tree_real(sample, DT))

        # calculate accuracy
        n_rows, n_cols = Y.shape
        n_correct = 0
        for i in range(n_rows):
            if predictions[i] == Y[i, 0]:
                n_correct += 1

        return n_correct / n_rows
        
################################################################################################
# FUNCTION: DT_train_binary_best
# PRECONDITION: two training lists and two validation lists, features and labels respectively 
# POSTCONDITION: A decision tree model will be returned (Class: BinaryTree)
################################################################################################
def DT_train_binary_best(X_train, Y_train, X_val, Y_val):

    minDepthTrain = len(X_train[0])
    maxDepthTrain = len(Y_train)                        
    depthSize = maxDepthTrain - minDepthTrain               # the amount of trees to construct 
    howAccurate = 0                                         # temp storage for the value of Test_binary
    returnIndex = 0                                         # the index of the treee to return
    treeIndex = 0
    DT = []

    while treeIndex < depthSize:                            # build a tree for every depth value from min to max
        DT.append(DT_train_binary(X_train, Y_train, maxDepthTrain - treeIndex))
                                                            # test each tree with the validation data
        if( DT_test_binary(X_val, Y_val, DT[treeIndex]) > howAccurate):
                                                            # if it is more "accurate(entropic); reassign the highest accuracy"
            howAccurate = DT_test_binary(X_val, Y_val, DT[treeIndex])
            returnIndex = treeIndex                         # store the index for returning
        treeIndex += 1

    return DT[returnIndex]                                  # return the best tree


################################################################################################
# FUNCTION: DT_train_real_best
# PRECONDITION: two training lists and two validation lists, features and labels respectively 
# POSTCONDITION: A decision tree model will be returned (Class: BinaryTree)
################################################################################################
def DT_train_real_best(X_train, Y_train, X_val, Y_val):

    minDepthTrain = len(X_train[0])
    maxDepthTrain = len(Y_train)                        
    depthSize = maxDepthTrain - minDepthTrain               # the amount of trees to construct 
    howAccurate = 0                                         # temp storage for the value of Test_binary
    returnIndex = 0                                         # the index of the treee to return
    treeIndex = 0
    DT = []                                                 


    while treeIndex < depthSize:                            # build a tree for every depth value from min to max
        DT.append(DT_train_real(X_train, Y_train, maxDepthTrain - treeIndex))
                                                            # test each tree with the validation data
        if( DT_test_real(X_val, Y_val, DT[treeIndex]) > howAccurate):
                                                            # if it is more "accurate(entropic); reassign the highest accuracy"
            howAccurate = DT_test_real(X_val, Y_val, DT[treeIndex])
            returnIndex = treeIndex                         # store the index for returning
        treeIndex += 1

    return DT[returnIndex]                                  # return the best tree
