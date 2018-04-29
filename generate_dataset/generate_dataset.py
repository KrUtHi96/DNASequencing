from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import random


class GenerateDataset:
    def __init__(self, error_rate, mutation_rate):

        self.scale = 3
        self.extreme = 10
        self.genes = ['A', 'G', 'T', 'C']

        self.error_rate = error_rate
        self.mutation_rate = mutation_rate

        self.genome = ''
        self.reads = []

    def int_to_gene(self, n):
        if -10 <= n < -2.5:
            return 'A'
        elif -2.5 <= n < 0:
            return 'G'
        elif 0 <= n < 2.5:
            return 'T'
        else:
            return 'C'

    def induce_error(self, read):

        for i in range(int(len(read) * self.mutation_rate)):
            random_index = random.randint(0, len(read) - 1)

            rn = random.random()

            if rn < self.error_rate:
                read = read[:random_index] + random.choice(
                    [c for c in self.genes if self.genes != read[random_index]]) + read[random_index + 1:]

        return read

    def random_genome(self, length=1000):

        a = -self.extreme / self.scale
        b = self.extreme / self.scale

        numbers = truncnorm(a=a, b=b, scale=self.scale).rvs(size=length)
        numbers = numbers.round().astype(int)

        # plt.hist(numbers, 2 * self.extreme + 1)
        # plt.show()

        self.genome = ''.join([self.int_to_gene(n) for n in numbers])

        return self.genome

    def random_reads(self, length=20, num=5):

        for i in range(num):
            start = random.randint(0, len(self.genome) - length)

            read = self.genome[start:start + length]

            read = self.induce_error(read)

            self.reads.append(read)

        return self.reads


if __name__ == '__main__':

    gd = GenerateDataset(0.8, 0.6)

    genome = gd.random_genome(length=5000)

    reads = gd.random_reads(length=50, num=10)

    count = 1
    for read in reads:
        print("Read", count, read)
        count += 1
