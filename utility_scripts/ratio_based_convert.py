import numpy as np
# an average filter convertor
LABEL_NROMAL = 0
LABEL_FALL = 1

WINDOW = 10
RIGHT_ADD = 5

CONVERT_RATIO = 0.4
NOT_CONVERT_RATIO = 0.6

result_label = [1,0,1,1,0,1]

def label_ratio(index, label_list, delta_window, left_add=0, right_add=0):
    di = np.abs(np.floor(delta_window / 2))
    label = label_list[index]
    if di + left_add < index < len(label_list) - di - right_add:
        return np.sum(np.array(label_list[int(index-di-left_add):int(index+di+right_add+1)]) == label) / (2 * di + left_add + right_add + 1)
    elif index <= di + left_add:
        return np.sum(np.array(label_list[:int(index+di+right_add)]) == label) / (index + di + right_add)
    else:
        return np.sum(np.array(label_list[int(index-di-left_add):]) == label) / (len(label_list) - index + di + left_add)

def main():
    for i in range(len(result_label)):
        if label_ratio(i, result_label, WINDOW) < CONVERT_RATIO:
            result_label[i] = LABEL_NROMAL if result_label[i] == LABEL_FALL else LABEL_FALL
        elif label_ratio(i, result_label, WINDOW) < NOT_CONVERT_RATIO:
            if label_ratio(i, result_label, WINDOW, right_add=RIGHT_ADD) < CONVERT_RATIO:
                result_label[i] = LABEL_NROMAL if result_label[i] == LABEL_FALL else LABEL_FALL
            else:
                pass
    
    print("RESULT: ", result_label)
if __name__ == "__main__":
    main()