import random
import time
import numpy as np
import os
import math
import sys
from scipy.stats import entropy as scipy_entropy
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn.random_projection import GaussianRandomProjection

os.environ["JOBLIB_TEMP_FOLDER"] = r"C:\temp"


def reliefFScore(X, y, **kwargs):
    if "k" not in kwargs.keys():
        k = 5
    else:
        k = kwargs["k"]
    n_samples, n_features = X.shape
    distance = pairwise_distances(X, metric='manhattan')
    score = np.zeros(n_features)
    for idx in range(n_samples):
        near_hit = []
        near_miss = dict()
        self_fea = X[idx, :]
        c = np.unique(y).tolist()
        stop_dict = dict()
        for label in c:
            stop_dict[label] = 0
        del c[c.index(y[idx])]
        p_dict = dict()
        p_label_idx = float(len(y[y == y[idx]])) / float(n_samples)
        for label in c:
            p_label_c = float(len(y[y == label])) / float(n_samples)
            p_dict[label] = p_label_c / (1 - p_label_idx)
            near_miss[label] = []
        distance_sort = []
        distance[idx, idx] = np.max(distance[idx, :])
        for i in range(n_samples):
            distance_sort.append([distance[idx, i], int(i), y[i]])
        distance_sort.sort(key=lambda x: x[0])
        for i in range(n_samples):
            if distance_sort[i][2] == y[idx]:
                if len(near_hit) < k:
                    near_hit.append(distance_sort[i][1])
                elif len(near_hit) == k:
                    stop_dict[y[idx]] = 1
            else:
                if len(near_miss[distance_sort[i][2]]) < k:
                    near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                else:
                    if len(near_miss[distance_sort[i][2]]) == k:
                        stop_dict[distance_sort[i][2]] = 1
            stop = True
            for (key, value) in stop_dict.items():
                if value != 1:
                    stop = False
            if stop:
                break
        near_hit_term = np.zeros(n_features)
        for ele in near_hit:
            near_hit_term = np.array(abs(self_fea - X[ele, :])) + np.array(near_hit_term)
        near_miss_term = dict()
        for (label, miss_list) in near_miss.items():
            near_miss_term[label] = np.zeros(n_features)
            for ele in miss_list:
                near_miss_term[label] = np.array(abs(self_fea - X[ele, :])) + np.array(near_miss_term[label])
            score += near_miss_term[label] / (k * p_dict[label])
        score -= near_hit_term / k
    return score


def top_select(score, percentage=0.05):
    score = np.ndarray.flatten(score)
    select_length = int(percentage * len(score))
    sorted_index = np.argsort(-score)
    top_index = sorted_index[0:select_length]
    return top_index


def processData(dataset):
    rawData = np.loadtxt(dataset, delimiter=',', encoding='utf-8-sig')
    X = rawData[:, :-1]
    Y = rawData[:, -1]
    minMAX = preprocessing.MinMaxScaler()
    X = minMAX.fit_transform(X)
    if X.shape[1] >= 7000:
        Score = reliefFScore(X, Y)
        top_index = top_select(Score)
        X = X[:, top_index]
    return X, Y


def callFitness(pop):
    popNum = pop.shape[0]
    tempFitness = np.zeros(popNum)
    threshold = 0.6
    for i in range(popNum):
        xIndex = np.where(pop[i] >= threshold)[0]
        if len(xIndex) == 0:
            tempFitness[i] = 0
        else:
            xTest = feat[:, xIndex]
            clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='manhattan', n_jobs=-1)
            kf = KFold(n_splits=5, shuffle=True)
            with np.errstate(invalid='ignore'):
                scores = cross_val_score(clf, xTest, label, cv=kf, n_jobs=-1)
                tempFitness[i] = np.mean(scores)
    return tempFitness


def popInitialization(NP, size, seed=None):
    np.random.seed(seed)
    return np.random.rand(NP, size)


def feature_entropy(population):
    entropy = np.zeros(population.shape[1])
    for dim in range(population.shape[1]):
        values = population[:, dim]
        hist, bin_edges = np.histogram(values, bins=10, range=(0, 1))
        hist = hist / hist.sum()
        hist = np.clip(hist, 1e-10, 1)
        entropy[dim] = scipy_entropy(hist, base=2)
    return np.maximum(entropy, 0)


def countFeatures(X):
    num = 0
    for d in range(len(X)):
        if X[d] >= 0.6:
            num = num + 1
    return num


def Mutation(pop, elite, gen, maxGen):
    mutationNum = int(Pm * len(pop))
    pop_mutated = np.zeros((mutationNum, size))
    adaptive_factor = 1.0 - (gen / maxGen)
    chaos_strength = 0.5 * adaptive_factor + 0.05
    quantum_tunnel = 0.3 * (1 - adaptive_factor)
    for i in range(mutationNum):
        idx = np.random.randint(0, len(pop))
        solution = pop[idx].copy()
        if np.random.rand() < quantum_tunnel:
            r = np.random.rand(size)
            solution = np.where(r < 0.2, elite, solution)
            solution = np.where(r > 0.8, np.random.rand(), solution)
        else:
            chaos = np.zeros(size)
            chaos[0] = np.random.rand()
            for j in range(1, size):
                chaos[j] = 3.99 * chaos[j - 1] * (1 - chaos[j - 1])
            active_dims = np.where(solution >= 0.6)[0]
            if len(active_dims) == 0:
                active_dims = np.random.choice(size, size // 10, replace=False)
            for dim in active_dims:
                elite_guidance = 0.3 * (elite[dim] - solution[dim])
                chaotic_explore = chaos_strength * (2 * chaos[dim] - 1)
                levy_step = np.random.standard_cauchy() * 0.1 * adaptive_factor
                solution[dim] += elite_guidance + chaotic_explore + levy_step
        solution = np.clip(solution, 0, 1)
        solution = np.where(np.abs(solution - 0.5) < 0.1,
                            np.random.choice([0, 1], size, p=[0.5, 0.5]),
                            solution)
        pop_mutated[i] = solution
    return pop_mutated


def pairing_strategy(population, fitness):
    n_pop, n_dim = population.shape
    active_rates = np.mean(population > 0.6, axis=0)
    proj_weights = 1 / (1 + np.exp(-5 * (np.abs(active_rates - 0.5) - 0.1)))
    max_proj_dim = min(200, max(50, int(n_dim * 0.1)))
    proj_dims = min(max_proj_dim, n_dim)

    if n_dim > 100:
        proj_indices = np.random.choice(
            n_dim,
            size=proj_dims,
            p=proj_weights / proj_weights.sum(),
            replace=False
        )
    else:
        proj_indices = np.arange(n_dim)

    proj_pop = population[:, proj_indices]
    centroid = np.mean(proj_pop, axis=0)
    dist_to_center = np.linalg.norm(proj_pop - centroid, axis=1)
    gravity_factor = dist_to_center / (np.max(dist_to_center) + 1e-10)

    sorted_idx = np.argsort(fitness)[::-1]
    pair_list = []
    paired_indices = set()
    diversity_boost = 0.2 * np.std(gravity_factor)

    for i in sorted_idx:
        if i in paired_indices:
            continue


        progress = gen / maxGen
        if progress < 0.5:

            sigmoid_progress = 1 / (1 + math.exp(-8 * (progress - 0.6)))
        else:

            sigmoid_progress = 1 / (1 + math.exp(-6 * (progress - 0.7)))


        dynamic_threshold = 0.03 + (0.12 - 0.03) * sigmoid_progress + diversity_boost


        dynamic_threshold = min(dynamic_threshold, 0.15)

        if gravity_factor[i] < dynamic_threshold:
            candidate_pool = [j for j in sorted_idx
                              if j not in paired_indices and j != i]

            if not candidate_pool:
                break

            similarities = np.sum(np.abs(proj_pop[i] - proj_pop[candidate_pool]), axis=1)
            partner_idx = candidate_pool[np.argmin(similarities)]

        else:
            candidate_pool = [j for j in sorted_idx
                              if j not in paired_indices and j != i]

            if not candidate_pool:
                break

            complement_scores = np.abs(gravity_factor[candidate_pool] - (1 - gravity_factor[i]))
            partner_idx = candidate_pool[np.argmin(complement_scores)]

        pair_list.append((i, partner_idx))
        paired_indices.add(i)
        paired_indices.add(partner_idx)

        if len(paired_indices) >= n_pop - 1:
            break

    unpaired = [j for j in range(n_pop) if j not in paired_indices]
    if unpaired:
        np.random.shuffle(unpaired)
        for i in range(0, len(unpaired) - 1, 2):
            pair_list.append((unpaired[i], unpaired[i + 1]))

    return pair_list


def reproduce(pop, fitness, flag, gen, maxGen):
    popNum = pop.shape[0]
    size = pop.shape[1]

    selected_pairs = pairing_strategy(pop, fitness)

    offPop = np.zeros((0, size))
    motherPop = np.zeros((0, size))

    for (p1, p2) in selected_pairs:
        alpha = np.random.rand(size)

        elite_mask = np.random.rand(size) < 0.1
        alpha[elite_mask] = 0 if fitness[p1] > fitness[p2] else 1

        off1 = alpha * pop[p1] + (1 - alpha) * pop[p2]
        off2 = (1 - alpha) * pop[p1] + alpha * pop[p2]

        off1 = np.clip(off1, 0, 1).reshape(1, -1)
        off2 = np.clip(off2, 0, 1).reshape(1, -1)

        offPop = np.vstack((offPop, off1, off2))

        if flag:
            motherPop = np.vstack((motherPop, pop[p1].reshape(1, -1), pop[p2].reshape(1, -1)))
        else:
            mother = p1 if fitness[p1] > fitness[p2] else p2
            motherPop = np.vstack((motherPop, pop[mother].reshape(1, -1)))

    if offPop.shape[0] < popNum:
        needed = popNum - offPop.shape[0]
        additional = popInitialization(needed, size)
        offPop = np.vstack((offPop, additional))

    if flag is False:
        crNum = int(offPop.shape[0] * Cr)
        finalFitness = callFitness(offPop)
        finalSort = np.argsort(-finalFitness)
        offPop = offPop[finalSort[:popNum - crNum]]

    finalPop = np.vstack((offPop, motherPop))
    return finalPop


class ParameterAdjustmentStrategy:
    def __init__(self, init_params, mem_size=5, adapt_rate=0.1):
        self.pm, self.cr, self.pr = init_params
        self.mem_size = mem_size
        self.adapt_rate = adapt_rate
        self.fitness_history = []
        self.diversity_history = []
        self.feature_history = []
        self.stagnation_count = 0
        self.phase = "exploration"
        self.best_fitness_record = []
        self.exploration_restart_count = 0

    def step(self, best_fitness, diversity, feature_count, gen, max_gen):
        self.fitness_history.append(best_fitness)
        self.diversity_history.append(diversity)
        self.feature_history.append(feature_count)
        self.best_fitness_record.append(best_fitness)

        if len(self.fitness_history) > self.mem_size:
            self.fitness_history.pop(0)
            self.diversity_history.pop(0)
            self.feature_history.pop(0)

        self._detect_phase(gen, max_gen)
        self._adapt_parameters(gen, max_gen)

        return self.pm, self.cr, self.pr

    def _detect_phase(self, gen, max_gen):
        if gen < max_gen * 0.3:
            self.phase = "exploration"
            return

        if len(self.fitness_history) >= self.mem_size:
            recent_improvement = max(self.fitness_history) - min(self.fitness_history)
            if recent_improvement < 0.005:
                self.stagnation_count += 1
            else:
                self.stagnation_count = max(0, self.stagnation_count - 1)

        if gen < max_gen * 0.7:
            self.phase = "exploitation"
            if self.stagnation_count > 3:
                self.phase = "re-exploration"
            return


        if gen >= max_gen * 0.7:

            if self.stagnation_count > 2 and self.exploration_restart_count < 2:
                self.phase = "re-exploration"
                self.exploration_restart_count += 1
                return


            if len(self.best_fitness_record) > 10:
                recent_best = self.best_fitness_record[-10:]
                if max(recent_best) - min(recent_best) < 0.005:
                    self.phase = "plateau_escape"
                    return

            self.phase = "refinement"

    def _adapt_parameters(self, gen, max_gen):
        if self.phase == "exploration":
            self.pm = np.clip(self.pm * 1.1, 0.3, 0.7)
            self.cr = np.clip(self.cr * 0.9, 0.4, 0.8)
            self.pr = np.clip(self.pr * 1.05, 0.7, 0.9)

        elif self.phase == "exploitation":
            avg_diversity = np.mean(self.diversity_history)
            if avg_diversity < 2.0:
                self.pm = np.clip(self.pm * 1.05, 0.2, 0.6)
                self.cr = np.clip(self.cr * 0.95, 0.5, 0.7)
            else:
                self.pm = np.clip(self.pm * 0.95, 0.1, 0.4)
                self.cr = np.clip(self.cr * 1.05, 0.6, 0.9)
            self.pr = np.clip(self.pr, 0.6, 0.8)

        elif self.phase == "re-exploration":

            self.pm = np.clip(0.6 + np.random.rand() * 0.2, 0.6, 0.8)
            self.cr = np.clip(0.2 + np.random.rand() * 0.2, 0.2, 0.4)
            self.pr = np.clip(0.6 + np.random.rand() * 0.2, 0.6, 0.8)

        elif self.phase == "plateau_escape":

            self.pm = np.clip(0.7 + np.random.rand() * 0.2, 0.7, 0.9)
            self.cr = np.clip(0.1 + np.random.rand() * 0.2, 0.1, 0.3)
            self.pr = np.clip(0.5 + np.random.rand() * 0.2, 0.5, 0.7)

            self.stagnation_count = 0

        elif self.phase == "refinement":

            avg_features = np.mean(self.feature_history)


            gen_progress = gen / max_gen
            exploration_factor = 0.3 * (1 - gen_progress)

            if avg_features > 100:

                self.pm = np.clip(self.pm * (1.1 + exploration_factor), 0.4, 0.7)
                self.cr = np.clip(self.cr * (0.9 - exploration_factor / 2), 0.5, 0.7)
            else:

                self.pm = np.clip(self.pm * (0.9 + exploration_factor), 0.3, 0.6)
                self.cr = np.clip(self.cr * (1.1 - exploration_factor / 2), 0.6, 0.8)


            self.pr = np.clip(0.7 - 0.2 * gen_progress, 0.5, 0.7)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SEED = int(sys.argv[1])
    else:
        SEED = None
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("SABWO")
    print("=" * 60)
    start_time = time.perf_counter()

    feat, label = processData("CLLSUB.csv")
    NP = 80
    maxGen = 100
    size = feat.shape[1]
    yVal = np.zeros(maxGen)
    bestVal = 0.0
    Pr = 0.8
    Pm = 0.4
    Cr = 0.5
    offNum = int(NP * Pr)

    initialPop = popInitialization(NP, size, seed=SEED)
    fitness = callFitness(initialPop)
    best_index = np.argmax(fitness)
    elite = initialPop[best_index].copy()
    bestVal = float(fitness[best_index])
    selectFeatures = countFeatures(elite)

    meta = ParameterAdjustmentStrategy([Pm, Cr, Pr])

    diversity_history = []
    gravity_distribution = []

    print("=" * 60)
    print(f"Dataset: {feat.shape[0]} samples x {feat.shape[1]} features")
    print("=" * 60)

    gen = 0
    while gen < maxGen:
        current_best_index = np.argmax(fitness)
        current_best_fitness = float(fitness[current_best_index])
        current_feature_count = countFeatures(initialPop[current_best_index])

        if (current_best_fitness > bestVal) or \
                (np.isclose(current_best_fitness, bestVal, atol=0.001) and (
                        current_feature_count < selectFeatures)):
            best_index = current_best_index
            elite = initialPop[best_index].copy()
            bestVal = current_best_fitness
            selectFeatures = current_feature_count

        sort = np.argsort(-fitness)
        pop1 = initialPop[sort[:offNum]]
        pop2 = reproduce(pop1, fitness[sort[:offNum]], False, gen, maxGen)
        pop3 = Mutation(pop2, elite, gen, maxGen)

        pop = np.append(pop2, pop3, axis=0)
        popFitness = callFitness(pop)
        popSort = np.argsort(-popFitness)

        initialPop = pop[popSort[:NP]]
        fitness = popFitness[popSort[:NP]]

        diversity = feature_entropy(initialPop)
        current_diversity = np.mean(diversity)
        diversity_history.append(current_diversity)

        Pm, Cr, Pr = meta.step(bestVal, current_diversity, selectFeatures, gen, maxGen)
        offNum = int(NP * Pr)

        active_rates = np.mean(initialPop > 0.6, axis=0)
        proj_weights = 1 / (1 + np.exp(-5 * (np.abs(active_rates - 0.5) - 0.1)))
        proj_indices = np.random.choice(
            size,
            min(100, size),
            p=proj_weights / proj_weights.sum(),
            replace=False
        )
        proj_pop = initialPop[:, proj_indices]
        centroid = np.mean(proj_pop, axis=0)
        dist_to_center = np.linalg.norm(proj_pop - centroid, axis=1)
        gravity_factor = dist_to_center / (np.max(dist_to_center) + 1e-10)
        gravity_distribution.append(gravity_factor)

        if gen < 50:
            display_threshold = 0.08
        else:
            display_threshold = 0.12

        core_percent = np.mean(gravity_factor < display_threshold) * 100
        edge_percent = np.mean(gravity_factor >= display_threshold) * 100

        print(f"Gen {gen + 1:03d} | Acc: {bestVal * 100:.2f}% | Feat: {selectFeatures} | "
              )

        yVal[gen] = bestVal * 100
        if len(pop1) <= 1:
            break
        gen += 1

    end_time = time.perf_counter()
    print("\n" + "=" * 60)
    print("Single Run Result")
    print("=" * 60)
    print(f"Best Accuracy: {bestVal * 100:.2f}%")
    print(f"Selected Features: {selectFeatures}")
    print(f"Running Time: {end_time - start_time:.2f} seconds")
    print("=" * 60)