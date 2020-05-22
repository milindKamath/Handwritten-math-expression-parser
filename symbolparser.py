import sys
from glob import iglob
from data_utility import *
import os
import pickle as pk


def read_in_data(file, path, ground_truth=False):
    """
    function to read in inkml data
    :param file: file containing a list of files to read in or all to be used when splitting
    :param path: path to upper level folder that contains all the training data
    :param ground_truth: boolean value determining if ground truth should be added
    :return: a list of Expressions
    """
    expersions = []
    if file == 'all':
        expersions = [inkml_to_Expression(path, f, ground_truth) for f in iglob(path+'/**/*.inkml', recursive=True)]
    elif os.path.splitext(file)[1][1:] == 'txt':
        with open(file) as o:
            expersions = [inkml_to_Expression(path, path + os.path.sep + f.strip(), ground_truth) for f in o]
    elif os.path.splitext(file)[1][1:] == 'inkml':
        return [inkml_to_Expression(path, file, ground_truth)]
    return list(filter(lambda x: x is not None, expersions))


def split(path, outfile):
    expressions = read_in_data('all', path, True)

    counts, symbol_to_index = get_symbol_counts(expressions)

    train_counts = [0 for i in range(len(counts))]
    train = []
    test_counts = [0 for i in range(len(counts))]
    test = []

    def sort_key(elem):
        return min(list(map(lambda x: counts[symbol_to_index[x.classification]], elem.symbols)))
    sort_expressions = expressions.copy()
    sort_expressions.sort(key=sort_key)

    i = 0
    while len(sort_expressions) > 0:
        expression = sort_expressions.pop()
        if i % 3 != 0:
            train.append(expression)
            for symbol in expression.symbols:
                train_counts[symbol_to_index[symbol.classification]] += 1
        else:
            test.append(expression)
            for symbol in expression.symbols:
                test_counts[symbol_to_index[symbol.classification]] += 1
        i += 1

    probability_train = count_to_probability(train_counts)
    probability_test = count_to_probability(test_counts)

    print("Kl divergence test to train:", kl_divergance(probability_test, probability_train))

    with open(outfile + '_train_data.txt', 'w') as o:
        for expression in train:
            o.write(expression.file + '\n')
    with open(outfile + '_test_data.txt', 'w') as o:
        for expression in test:
            o.write(expression.file + '\n')


def getRel(expressions):
    mean = []
    for symb in expressions.symbols:
        strokes = np.empty((0, 2))
        for strks in symb.strokes:
            strokes = np.append(strokes, strks.points, axis=0)
        mean.append([np.mean(strokes[:, 0]), np.mean(strokes[:, 1])])
    index = np.argsort(mean, axis=0)[:, 0]
    sym = {}
    for i, symb in enumerate(expressions.symbols):
        id = ''
        for stroke in symb.strokes:
            id += str(stroke.id) + ":"
        sym[i] = id
    name = []
    for i in range(len(index)-1):
        name.append(sym[index[i]] + ', ' + sym[index[i+1]])
    return name


def lg_or_outputBaseline(expression, path):
    filepath = path + expression.file.replace('inkml', 'lg')
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(filepath, 'w') as o:
        o.write("# IUD, \"%s\"\n" % expression.uid)
        o.write("# Objects(%d):\n" % len(expression.symbols))
        for symbol in expression.symbols:
            name = ''
            strokes = ''
            for stroke in symbol.strokes:
                name += '%d:' % stroke.id
                strokes += ', %d' % stroke.id
            o.write('O, %s, %s, %.1f%s\n' % (name, symbol.classification, symbol.weight, strokes))
        rel = getRel(expression)
        o.write("\n")
        o.write("# Relations from SRT:\n")
        if len(expression.symbols) == 1:
            o.write('')
        else:
            for r in rel:
                o.write('R, %s, Right, 1.0\n' % r)


def evaluate_for_all(file, path, outpath, parse):
    # expressions = read_in_data(file, path, True)
    # exp = open(file+'GTexpressions.pickle', 'wb')
    # pk.dump(expressions, exp)
    # exp.close()
    exp = open(file+'GTexpressions.pickle', 'rb')
    expressions = pk.load(exp)
    for expression in expressions:
        parse(expression, outpath)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage [Python] [split/create/train/evaluate]')
        exit(1)
    if sys.argv[1] == 'split':
        if len(sys.argv) == 4:
            split(sys.argv[2], sys.argv[3])
        else:
            print('Usage [Python] split [path to data] [output file]')
            # split C:\Users\milin\PycharmProjects\PRProject2\Train-Revised\inkml\ revised
    if sys.argv[1] == 'evaluate':
        if 5 <= len(sys.argv) <=6:
            parse_type = sys.argv[2]
            file = sys.argv[3]
            if len(sys.argv) == 5:
                path = ''
                outpath = sys.argv[4]
            else:
                path = sys.argv[4]
                outpath = sys.argv[5]
            parse = None
            if parse_type.lower() == 'baseline':
                parse = lg_or_outputBaseline

            evaluate_for_all(file, path, outpath, parse)
        else:
            print('Usage [Python] evaluate [segmentor] [Bulk file] [path to data] [path output]')
            # evaluate baseline test_train_data.txt C:\Users\milin\PycharmProjects\PRProject2\Train\inkml\ C:\Users\milin\PycharmProjects\PRProject2\Parse\baseline\