import numpy as np
from scipy.stats import mode

DEBUG = False
COLUMNS = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
           "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
           "Mitoses", "Class"]
MY_COLUMNS = [7, 3, 4, 5, 8]
BENIGN = 2  # Code for benign in Class column
MALIGNANT = 4  # Code for malignant in Class column


def load_dataset(fname: str) -> np.array:
    with open(fname, 'r') as f:
        text = f.read()
    # parse csv as list of lists
    lol = [line.split(',') for line in tuple(text.split("\n"))[:-1]]
    arr = np.asarray(lol)
    arr = arr[(arr != '?').all(axis=1)]  # remove rows that contain '?'
    return arr.astype(int)


def binary_entropy(data):
    count = len(data)
    p0 = sum(b[-1] == 2 for b in data) / count
    if p0 == 0 or p0 == 1: return 0
    p1 = 1 - p0
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def infogain(data, fea, threshold):  # x_fea <= threshold;  fea = 2,3,4,..., 10; threshold = 1,..., 9
    count = len(data)
    d1 = data[data[:, fea - 1] <= threshold]
    d2 = data[data[:, fea - 1] > threshold]
    if len(d1) == 0 or len(d2) == 0:
        return 0
    return binary_entropy(data) - (len(d1) / count * binary_entropy(d1) + len(d2) / count * binary_entropy(d2))


def q3(d: np.array) -> (int, int, int, int):
    fea = COLUMNS.index("Mitoses")
    threshold = 2
    c = len(d)
    below = d[d[:, fea] < threshold]
    bb = len(below[below[:, -1] == BENIGN])
    mb = len(below) - bb
    above = d[d[:, fea] >= threshold]
    ba = len(above[above[:, -1] == MALIGNANT])
    ma = len(above) - ba
    return ba, bb, ma, mb


# def info_gain(d: np.array, fea, threshold) -> np.float64:
#     hy = binary_entropy(d)
#     hyx = 0
#     for kx in range(1, 11):
#         x = d[d[:,fea] == kx]
#         px = len(x) / len(d)
#         if px == 0:
#             continue
#         hyxx = 0
#         for ky in [BENIGN, MALIGNANT]:
#             yx = x[x[:,-1] == ky]
#             pyx = len(yx) / len(x)
#             if pyx == 0:
#                 continue
#             hyxx += pyx * np.log2(pyx)
#         hyxx *= -1
#         hyx += px * hyxx
#     return hy - hyx


class Node:
    def __init__(self, fea, threshold):
        self.fea = fea
        self.threshold = threshold
        self.left = None
        self.right = None
        self.data = None


def write_p2(d) -> None:
    ig = [[infogain(d, fea, t) for t in range(1,10)] for fea in MY_COLUMNS]
    ig = np.array(ig)
    ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    root = Node(MY_COLUMNS[ind[0]], ind[1] + 1)
    create_tree(d, root)
    with open('P2.txt', 'w') as f:
        f.write("Outputs:\n")
        f.write("@id\n")
        f.write("pjmatthews\n")
        f.write("@answer_1\n")
        f.write(f"{len(d[d[:, -1] == 2])},{len(d[d[:, -1] == 4])}\n")
        f.write("@answer_2\n")
        f.write(f"{binary_entropy(d)}\n")
        f.write("@answer_3\n")
        a3 = q3(d)
        f.write(f'{str(a3[0])},{str(a3[1])},{str(a3[2])},{str(a3[3])}\n')
        f.write("@answer_4\n")
        f.write(f'{infogain(d, 10, 2)}\n')
        f.write("@tree_full\n")
        f.write(f"{print_tree(root)}\n")
        f.write("@answer_6\n")
        f.write(f"{get_height(root)}\n")
        f.write("@label_full\n")
        f.write(f'{",".join([str(i[0]) for i in predict_all(load_dataset("test.txt"), root)])}\n')
        f.write("@tree_pruned\n")
        prune_tree(5, root)
        f.write(f"{print_tree(root)}\n")
        f.write("@label_pruned\n")
        f.write(f'{",".join([str(i[0]) for i in predict_all(load_dataset("test.txt"), root)])}\n')
        f.write("@answer_10\n")
        f.write("None\n")


def find_best_split(data):
    c = len(data)
    c0 = sum(b[-1] == 2 for b in data)
    if c0 == c: return 2, None
    if c0 == 0: return 4, None
    ig = [[infogain(data, f, t) for t in range(1, 10)] for f in MY_COLUMNS]
    ig = np.array(ig)
    max_ig = max(max(i) for i in ig)
    if max_ig == 0:
        if c0 >= c - c0:
            return (2, None)
        else:
            return (4, None)
    ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    fea, threshold = MY_COLUMNS[ind[0]], ind[1] + 1
    return fea, threshold


def split(data, node):
    fea, threshold = node.fea, node.threshold
    d1 = data[data[:, fea-1] <= threshold]
    d2 = data[data[:, fea-1] > threshold]
    return d1,d2


def create_tree(data, node):
    d1,d2 = split(data, node)
    f1, t1 = find_best_split(d1)
    f2, t2 = find_best_split(d2)
    if t1 is None:
        node.left = f1
    else:
        node.left = Node(f1,  t1)
        node.left.data = d1
        create_tree(d1, node.left)
    if t2 is None:
        node.right = f2
    else:
        node.right = Node(f2, t2)
        node.right.data = d2
        create_tree(d2, node.right)


def _print_tree(tabs, root: Node):
    string = ""
    string += "\t" * tabs
    if type(root.left) is int:
        string += f"if (x{root.fea} <= {root.threshold}) return {root.left}\n"
    else:
        string += f"if (x{root.fea} <= {root.threshold})\n"
        string += _print_tree(tabs+1, root.left)
    string += "\t" * tabs
    if type(root.right) is int:
        string += f"else return {root.right}\n"
    else:
        string += f"else\n"
        string += _print_tree(tabs+1, root.right)
    return string


def print_tree(root: Node) -> str:
    return "" + _print_tree(0, root)


def _get_height(current: int, root: Node) -> int:
    if type(root.left) is int:
        left = current + 1
    else:
        left = _get_height(current + 1, root.left)
    if type(root.right) is int:
        right = current + 1
    else:
        right = _get_height(current + 1, root.right)
    return max(left, right)


def get_height(root: Node) -> int:
    return _get_height(0, root)


def predict(x: np.array, root: Node) -> int:
    while type(root) is not int:
        if x[root.fea-1] <= root.threshold:
            root = root.left
        else:
            root = root.right
    return root


def predict_all(X: np.array, root: Node) -> np.array:
    predictions = np.zeros((len(X), 1), dtype=int)
    for i, row in enumerate(X):
        predictions[i] = predict(row, root)
    return predictions


def _prune_tree(max_height: int, current_height: int, root: Node):
    if current_height == max_height:
        return int(mode(root.data[:,-1]).mode[0])
    else:
        if type(root.left) is not int:
            root.left = _prune_tree(max_height, current_height+1, root.left)
        if type(root.right) is not int:
            root.right = _prune_tree(max_height, current_height+1, root.right)
        return root


def prune_tree(height: int, root: Node):
    _prune_tree(height, 0, root)



def main() -> None:
    d = load_dataset("breast-cancer-wisconsin.data")
    # ig = [[infogain(d, fea, t) for t in range(1,10)] for fea in MY_COLUMNS]
    # ig = np.array(ig)
    # ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    # root = Node(MY_COLUMNS[ind[0]], ind[1] + 1)
    # root.data = d
    # create_tree(d, root)
    # prune_tree(5, root)
    pass
    if not DEBUG:
        write_p2(d)
        pass


if __name__ == "__main__":
    main()
    exit(1)
