import matplotlib.pyplot as plt
import time
import random

from dna_utils import helper_functions


class ShinglesScore:

    def __init__(self, n=12):

        self.n = n
        self.score = 0

    def shingles_set(self, genome):

        S = set()
        N = len(genome)

        for i in range(N - self.n + 1):
            S.add(genome[i: i + self.n])

        return S

    def ng_score(self, genome_a, genome_b):

        S_a, S_b = self.shingles_set(genome_a), self.shingles_set(genome_b)

        self.score = len(S_a.intersection(S_b))
        self.score /= max(len(S_a), len(S_b))

        return self.score

    def dot_plot_matrix(self, genome_a, genome_b):
        print("Dot Plot Matrix...")

        rows, cols = len(genome_a) - self.n + 1, len(genome_b) - self.n + 1

        # matrix = [[0] * cols for i in range(rows)]
        plt.title("Similarity Between Genomes " + str(self.score))

        for i in range(min(rows, cols)):
            if genome_a[i: i + self.n] == genome_b[i: i + self.n]:
                # print(i)
                plt.plot(i, i, 'bo', markersize=0.5)

        plt.xlim(0, cols)
        plt.ylim(0, rows)

        plt.xlabel("Genome B" + " (" + "Length : " + str(cols + self.n - 1) + ")")
        plt.ylabel("Genome A" + " (" + "Length : " + str(rows + self.n - 1) + ")")

        plt.rcParams["figure.figsize"] = (10, 10)

        plt.show()


def main():
    genome = helper_functions.read_genome("../Dataset/lambda_virus.fa")
    a, b = genome[:1000], genome[:1000]

    for i in range(20):
        index = random.randint(0, len(b))
        b = b[:index] + 'C' + b[index + 1:]

    # b = a




    print(len(a), len(b))

    start = time.time()

    ss = ShinglesScore()
    score = ss.ng_score(a, b)

    print("Global Similarity Score", score)
    ss.dot_plot_matrix(a, b)

    end = time.time()

    print("Time for Execution: ", end - start)


if __name__ == '__main__':
    main()
