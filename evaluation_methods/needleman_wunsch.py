# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:19:05 2018

@author: KRUTHI
"""

import numpy as np


def similarity(a, b):
    return int(a == b)


def preprocess_text(a, b):
    return '$' + a, '$' + b


def align(a, rows, b, columns, gap_penalty, F):
    align_a, align_b = '', ''

    i = rows - 1
    j = columns - 1

    while (i > 0) or (j > 0):

        if (i > 0 and j > 0 and F[i, j] == F[i - 1, j - 1] + similarity(a[i], b[j])):
            align_a = a[i] + align_a
            align_b = b[j] + align_b

            i, j = i - 1, j - 1

        elif (i > 0 and F[i, j] == F[i - 1, j] + gap_penalty):
            align_a = a[i] + align_a
            align_b = "-" + align_b

            i = i - 1

        else:
            align_a = "-" + align_a
            align_b = b[j] + align_b

            j = j - 1
    return align_a, align_b


def NW_score(a, b, gap_penalty=-5):
    a, b = preprocess_text(a, b)

    rows = len(a)
    columns = len(b)

    F = np.zeros(shape=(rows, columns))

    for i in range(rows):
        F[i, 0] = gap_penalty * i

    for j in range(columns):
        F[0, j] = gap_penalty * j

    for i in range(1, rows):
        for j in range(1, columns):
            match = F[i - 1, j - 1] + similarity(a[i], b[j])

            delete = F[i - 1, j] + gap_penalty

            insert = F[i, j - 1] + gap_penalty

            F[i, j] = max(match, insert, delete)

    align_a, align_b = align(a, rows, b, columns, gap_penalty, F)

    score = F[rows - 1, columns - 1]

    return (align_a, align_b, score)


def test1(a, b):
    align_a, align_b, score = NW_score(a, b)

    print("String 1", a)
    print("String 2", b)

    print(align_a, align_b, sep="\n")

    print("Final Score : ", score)


if __name__ == "__main__":
    test1("TCGCTTAAGTATGCTTTTAG", "GAACAAAACTCAATAAAATAGAACAAAACTCAATAAAATA")
    test1("paper", "coinpaper")
