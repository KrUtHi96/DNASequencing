import random

from graphviz import render

from overlap_methods.suffix_tree import SuffixTree
from overlap_methods.suffix_tree import SuffixTreeNode
from generate_dataset import generate_dataset
from read_dataset import read_dataset


def naive_overlap(a, b, min_length=0):
    start = 0
    while True:
        start = a.find(b[:min_length], start)
        if start == -1:
            return 0
        if b.startswith(a[start:]):
            return len(a) - start
        start += 1


def verify_matrix(reads, mat, read_len, min_overlap=0, verbose=False):
    n = len(reads)
    in_correct = 0
    total = 0
    for prefix_of_read in range(n):
        for suffix_of_read in range(n):

            if prefix_of_read == suffix_of_read:
                continue

            total += 1
            score = int(mat[prefix_of_read][suffix_of_read])

            if verbose:
                print('Reads:')
                print(reads[prefix_of_read])
                print(reads[suffix_of_read])
                print('\nScore: {}\n\nAligned:'.format(score))
                print(reads[prefix_of_read][:score])
                print(reads[suffix_of_read][read_len - score:])

            if not reads[prefix_of_read][:score] == reads[suffix_of_read][read_len - score:]:
                true_score = naive_overlap(reads[suffix_of_read], reads[prefix_of_read], min_length=min_overlap)
                print('\nReads:')
                print(reads[prefix_of_read])
                print(reads[suffix_of_read])
                print('\nScore: {}\nTrue Score: {}\n\nAligned:'.format(score, true_score))
                print(reads[prefix_of_read][:score])
                print(reads[suffix_of_read][read_len - score:])

                score = true_score

                print(reads[prefix_of_read][:score])
                print(reads[suffix_of_read][read_len - score:])

                in_correct += 1
    print('{} / {} [{}%]'.format(in_correct, total, in_correct / total))


def run_testcase(input_strings, min_overlap=0, verbose=False, save_tree=False):
    SuffixTreeNode.new_identifier = 0
    suffix_tree = SuffixTree()

    for s in input_strings:
        suffix_tree.append_string(s)

    if save_tree:
        open('graph', 'wb+').write(suffix_tree.to_graphviz().encode())
        render('dot', 'png', 'graph')

    overlap_matrix = []
    for str_num, string in enumerate(input_strings):
        overlap_matrix.append(suffix_tree.overlap(str_num, string, min_overlap=min_overlap, verbose=verbose))

    return overlap_matrix


def testcase_1():
    min_overlap = 0
    read_len = 6

    input_strings = ['GACATA', 'ATAGAC', 'ATAGAC', 'GACATA', 'GACATA', 'ATAGAC', 'ATAGAC', 'ATAGAC', 'GACATA', 'GACATA',
                     'ATAGAC', 'ATAGAC', 'ATAGAC']

    mat = run_testcase(input_strings, min_overlap=min_overlap)
    print('Overlap Matrix Computed...')
    print('Verifying Matrix...')
    verify_matrix(input_strings, mat, read_len=read_len, min_overlap=min_overlap)


def testcase_2():
    min_overlap = 0
    read_len = 100

    gd = generate_dataset.GenerateDataset(error_rate=0.6, mutation_rate=0.6)
    _ = gd.random_genome(length=10000)
    input_strings = gd.random_reads(length=read_len, num=1000)

    mat = run_testcase(input_strings, min_overlap=min_overlap)
    print('Overlap Matrix Computed...')
    print('Verifying Matrix...')
    verify_matrix(input_strings, mat, read_len=read_len, min_overlap=min_overlap)


def testcase_3():
    min_overlap = 0
    read_len = 100

    input_strings, q = read_dataset.read_fastq('../Dataset/phix.fastq')

    for i, read in enumerate(input_strings):
        input_strings[i] = read.replace('N', random.choice('AGTC'))

    mat = run_testcase(input_strings, min_overlap=min_overlap)
    print('Overlap Matrix Computed...')
    print('Verifying Matrix...')
    verify_matrix(input_strings, mat, read_len=read_len, min_overlap=min_overlap)


if __name__ == '__main__':
    testcase_1()
    # testcase_2()
    # testcase_3()
