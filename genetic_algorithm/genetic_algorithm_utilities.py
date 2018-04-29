import random


class GeneticAlgorithmUtil:

    def __init__(self, reads, overlap_matrix):
        self.reads = reads
        self.overlap_matrix = overlap_matrix

        self.reads_n = len(reads)

    def add(self, i, j):
        offset = int(self.overlap_matrix[i, j])
        return self.reads[j][offset:]

    def generate_genome(self, index_list):
        genome = self.reads[index_list[0]]
        for i in range(1, len(index_list)):
            genome += self.add(index_list[i - 1], index_list[i])
        return genome

    def initialize_population(self, size):
        i = 0
        population = {}
        while i < size:
            index_list = []
            temp = list(range(self.reads_n))
            while temp:
                index = random.choice(temp)
                temp.remove(index)
                index_list.append(index)
            genome = self.generate_genome(index_list)
            if genome not in population:
                population[genome] = index_list
                i += 1
        return population

    def initialize_pop(self, size, gen_len):
        i = 0
        population = dict()
        while i < size:
            genome = ''
            index_list = []
            temp = list(range(self.reads_n))
            while len(genome) < gen_len:
                print('Temp:', len(temp))
                index = random.choice(temp)
                temp.remove(index)
                index_list.append(index)
                genome += self.reads[index]

            if genome not in population:
                population[genome] = index_list
                i += 1
        return population

    def fitness_score1(self, index_list):
        score = 0
        for i in range(len(index_list) - 1):
            score += self.overlap_matrix[index_list[i], index_list[i + 1]]
        return score

    def fitness_score2(self, index_list):
        score = 0
        for i in range(len(index_list) - 1):
            for j in range(len(index_list) - 1):
                score = score + (abs(i - j) * self.overlap_matrix[index_list[i], index_list[j]])
        return score

    def selection(self, population, n, fn):
        new_population = {}
        fitness = {}
        for i in population:
            fitness[i] = fn(population[i])
        ordered = sorted(fitness.items(), key=lambda x: x[1], reverse=True)
        print("ordered", len(ordered))
        for i in range(n):
            genome = ordered[i][0]
            new_population[genome] = population[genome]
        return new_population, fitness

    def crossover1(self, p1, p2, start, end):
        temp = p1[start: end + 1]
        count = 0
        for i in p2:
            if i not in temp:
                if count < start:
                    temp = [i] + temp
                else:
                    temp = temp + [i]
                count += 1
        temp = self.mutation_pop(temp)
        genome = self.generate_genome(temp)
        return genome, temp

    def mutation(self, index_list):
        t = len(index_list)
        a = random.randint(0, t - 1)
        b = random.randint(0, t - 1)
        index_list[a], index_list[b] = index_list[b], index_list[a]

        return index_list

    def mutation_pop(self, index_list):
        m_rate = int(0.2 * len(index_list))
        i = 0
        while i <= m_rate:
            # print("inside pop m ", i, m_rate)
            index = random.randint(0, len(index_list) - 1)
            t = random.randint(0, self.reads_n - 1)
            # print(t)
            if t not in index_list:
                index_list[index] = t
                i += 1
        # print("Mutation POP done !!")
        return self.mutation(index_list)

    def find_neighbours(self, index_list_1, index_list_2):
        neighbours = {}
        length_1 = len(index_list_1)
        length_2 = len(index_list_2)

        for i, idx in enumerate(index_list_1):
            if idx not in neighbours:
                neighbours[idx] = set()

            neighbours[idx].add(index_list_1[i - 1])
            neighbours[idx].add(index_list_1[(i + 1) % length_1])

        for i, idx in enumerate(index_list_2):
            if idx not in neighbours:
                neighbours[idx] = set()

            neighbours[idx].add(index_list_2[i - 1])
            neighbours[idx].add(index_list_2[(i + 1) % length_2])

        return neighbours

    def crossover_edge_recombination(self, p1, p2):
        length = len(p1)

        neighbour_list = self.find_neighbours(p1, p2)
        # print('Neighbour List:', neighbour_list)

        current_node = random.choice((p1[0], p2[0]))
        # print('Start node:', current_node)

        child = [current_node]
        while len(child) < length:
            # print("Inside while", len(child), length)
            # Remove selected node from neighbour_lists
            for node in neighbour_list:
                if current_node in neighbour_list[node]:
                    neighbour_list[node].remove(current_node)

            min_neigh_list = neighbour_list[current_node]
            del neighbour_list[current_node]

            if len(min_neigh_list) > 0:  # if the chosen node has any neighbours
                # get the best match out of neighbours as next
                max_overlap = self.overlap_matrix[
                    current_node, max(min_neigh_list, key=lambda x: self.overlap_matrix[current_node, x])]
                possibilities = list(
                    filter(lambda x: self.overlap_matrix[current_node, x] == max_overlap, min_neigh_list))
                current_node = possibilities[random.randint(0, len(possibilities) - 1)]
            else:
                # get the best match out of every node as next
                max_overlap = self.overlap_matrix[
                    current_node, max(neighbour_list, key=lambda x: self.overlap_matrix[current_node, x])]
                possibilities = list(
                    filter(lambda x: self.overlap_matrix[current_node, x] == max_overlap, neighbour_list))
                current_node = possibilities[random.randint(0, len(possibilities) - 1)]
            child.append(current_node)  # add the node to the solution
        return self.generate_genome(child), child


if __name__ == '__main__':
    u = GeneticAlgorithmUtil(['aaaaa', 'abbadbvad'], [[1, 2], [3, 4]])
    u.crossover_edge_recombination([1, 2, 3, 4, 5], [3, 4, 1, 2, 5])
