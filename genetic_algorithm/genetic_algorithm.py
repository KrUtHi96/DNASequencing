import random

import time

from generate_dataset import generate_dataset
from overlap_methods import overlap_score_pigeonhole
from evaluation_methods import shingles_score
import genetic_algorithm_utilities
from read_dataset import read_dataset
from dna_utils import helper_functions

import numpy as np

from overlap_methods.suffix_tree import SuffixTreeNode, SuffixTree


def GeneticAlgorithm(size=100, generations=60, select_n=60, threshold=0.94):
    gd = generate_dataset.GenerateDataset(error_rate=0.0, mutation_rate=0.0)
    genome = gd.random_genome(length=30000)

    reads = gd.random_reads(length=1000, num=5000)

    # genome = read_dataset.read_genome('../Dataset/ecoli.fa')
    # reads, _ = read_dataset.read_fastq('../Dataset/e_coli_1000.fq')

    print(len(reads[0]), len(genome))

    # Preprocessing reads
    # processed_reads = []
    # for i, read in enumerate(reads):
    #     current_read = read.replace('N', random.choice('AGTC'))
    #     processed_reads.append(current_read)
    #
    #     current_read_comp = helper_functions.reverse_complement(current_read)
    #     processed_reads.append(current_read_comp)
    #
    # reads = processed_reads
    # Pigeonhole principle
    # osp = overlap_score_pigeonhole.OverlapScorePigeonhole(reads, overlap_minimum=20, max_error=3)
    # overlap_matrix = osp.overlap_scores()

    # Construct Suffix tree
    SuffixTreeNode.new_identifier = 0
    suffix_tree = SuffixTree()

    for read in reads:
        suffix_tree.append_string(read)

    # Compute matrix
    overlap_matrix = []
    for str_num, string in enumerate(reads):
        overlap_matrix.append(suffix_tree.overlap(str_num, string, min_overlap=20))

    overlap_matrix = np.matrix(overlap_matrix)
    overlap_matrix = overlap_matrix.T

    print("************matrix computed********************")

    ss = shingles_score.ShinglesScore(n=12)

    gau = genetic_algorithm_utilities.GeneticAlgorithmUtil(reads, overlap_matrix)

    population = gau.initialize_pop(size, len(genome)) #Works..
    #population = gau.initialize_population(size)

    max_so_far = {'genome': '', 'score': -10}

    count = 1
    while count <= generations:

        for i in population:
            # print("population",len(i),"original", len(genome))
            score = ss.ng_score(i, genome)

            if score > max_so_far['score']:
                max_so_far['genome'], max_so_far['score'] = i, score
                print("Max Score so far :", max_so_far['score'])
            if score >= threshold:
                print("Genome Found! in generation", count, score)
                return i, score

        population, fitness = gau.selection(population, select_n, gau.fitness_score1)

        print("Selection done")

        while len(population) < size:
            # print(len(population), size)
            temp = list(population.values())
            a = random.choice(temp)
            b = random.choice(temp)
            if a != b:
                # print("CHeck", a, b)

                start = random.randint(1, len(a) - 2)
                end = random.randint(start + 1, len(a))
                # genome_new, index_list = gau.crossover1(a, b, start, end)

                genome_new, index_list = gau.crossover_edge_recombination(a, b)
                if genome_new not in population:
                    population[genome_new] = index_list

        print("Crossover done")

        # mutation
        for i in list(population.keys()):
            temp = gau.mutation_pop(population[i])
            temp_gen = gau.generate_genome(temp)
            if temp_gen not in population:
                population.pop(i)
                population[temp_gen] = temp

        print("Generation :", count, max_so_far['score'])
        count += 1

        print("Population len", len(population))

    return max_so_far['genome'], max_so_far['score']


if __name__ == '__main__':
    start = time.time()
    reconstructed_genome, score = GeneticAlgorithm(threshold=0.99)
    print('Total Time: {}'.format((time.time() - start)))

    print("Best Score :", score)
