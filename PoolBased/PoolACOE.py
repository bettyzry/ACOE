from __future__ import division
import numpy as np
import util
import evaluate


class PoolACOE:
    def __init__(self, ways=5, queryrate=0.005, itera=100, threshold=0.95, kpi=1, weight=None, seasonal=1440):
        self.queried = []
        self.ways = ways
        self.weight = weight or [2, 2, 2, 2, 2]
        self.query_num = None
        self.query_rate = queryrate
        self.itera = itera
        self.prethreshold = threshold
        self.truethreshold = 1 - (1 - threshold)
        self.kpi = kpi
        self.last_loss = 100
        self.per = max(1 - threshold, 0.000001)
        self.ano = 0
        self.seasonal = seasonal
        return

    def predict(self, score, y_true, dropstress=False):
        self.updata_weight(score, y_true)
        diff_value_normalize = util.getsum_score(score, self.weight)
        if dropstress:
            y_pre, y_stress, nostress_score = self.DropSeasonal(diff_value_normalize)
        else:
            y_pre = util.score2label_threshold(diff_value_normalize, self.prethreshold)
            y_stress = np.zeros(len(y_pre))
            nostress_score = np.copy(diff_value_normalize)
        return y_pre, y_stress, diff_value_normalize, nostress_score

    def updata_weight(self, score, label):
        self.query_num = int(len(score[0]) * self.query_rate)
        result = np.copy(label)
        loss = 2
        t = 0
        t2 = 0
        itera = [0]
        while abs(self.last_loss - loss) > 0.001 or t2 < 2:
            if abs(self.last_loss - loss) < 0.001:
                t2 += 1
            self.last_loss = loss
            t += 1
            itera.append(t)
            s, l = self.do_query(inputscore=score, label=result)
            bias = self.part_weight(s, l)
            loss = self.get_partbest_weight_w2(s, l, t, bias)
            loss = loss / len(self.queried)
            self.per = max(sum(l) / len(l), 0.000001)
            self.ano = sum(l)

            # diff_value_normalize = util.getsum_score(score, self.weight)
            # _, _, _, _, _, _, _, _, _, text = evaluate.evaluate('kpi', True, diff_value_normalize, label, self.prethreshold)
            # print(text)

    ########################  获取局部最优  ####################################
    def part_weight(self, score, label):
        from sklearn import metrics
        w = []
        if sum(label) == 0:
            w = [0.5] * self.ways
            return w
        for i in range(len(score)):
            auc_roc = metrics.roc_auc_score(label, score[i])
            w.append(auc_roc)
        w = util.norm_to_0and1(w)
        return w

    def get_partbest_weight_w2(self, s, l, itera, w):
        def get_random_weight(itera, w, self_weight):
            import random
            weight = []
            for i in range(len(w)):
                alpha = random.uniform(1, 1 + 10 / itera)
                mul_div = random.random()
                if mul_div <= w[i]:
                    weight.append(self_weight[i] * alpha)
                else:
                    weight.append(self_weight[i] / alpha)
            weight = util.norm_to_1(weight)
            return weight

        loss = [10000]
        count = 0
        last_loss = 0
        while abs(last_loss - min(loss)) / len(self.queried) > 0.001:
            last_loss = min(loss)
            weightlist = [self.weight]
            for i in range(200):
                weightlist.append(get_random_weight(itera + count, w, self.weight))
            loss = []
            for weight in weightlist:
                l1 = self.get_sum_auc(s, l, weight)
                loss.append(l1)
            index = loss.index(min(loss))
            partbest_weight = weightlist[index]
            self.weight = partbest_weight
            count += 1
        return min(loss)

    ############################################################
    def get_sum_auc(self, s, l, weight):
        def loss(l, s):
            from pandas import Series
            data = Series(s)
            rank_data = data.sort_values(ascending=True)
            index = rank_data.index
            sum = 0
            for i in range(len(index)):
                if l[index[i]] == 1:
                    sum += len(index) - i  # 1 - i/len(index)
                else:
                    sum += i
            return sum / len(index)

        s = util.getsum_score(s, weight)
        l1 = loss(l, s)
        return l1

    ######################  查询  ########################################
    def get_mult_marge_tobe_queried(self, score):
        # 随机获取num个数的索引
        # score: [,,,]
        def get_random_tobe_queried(start, end, query_num):
            # 随机获取num个数的索引
            # score: [,,,]
            import random
            index = []
            for i in range(query_num):
                num = random.randint(start, end)
                index.append(int(num))
            return list(index)

        margin = int(self.query_num * (self.truethreshold) ** (self.truethreshold / self.per))
        # print(1-self.per, self.threshold, margin, self.query_num, len(self.queried))
        mult = max(self.query_num - margin, 1)

        import math
        groupnum = max(1, math.ceil(mult / 10))
        import math
        sorted_array = np.sort(score)
        temp_array = score.tolist()

        index_q = math.floor(len(sorted_array) * self.truethreshold)
        if int(index_q + margin / 2) >= len(score):
            query = sorted_array[-margin:]
        else:
            query = sorted_array[int(index_q - margin / 2):int(index_q + margin / 2)]
        # if len(query) != margin:
        #     raise ValueError("len(margin) != self.query_num", len(query), margin, int(index_q + margin))

        for i in range(10):
            index_q = math.floor(len(sorted_array) * i / 10)
            if int(index_q + groupnum) >= len(score):
                random_index = get_random_tobe_queried(index_q, len(score) - 1, groupnum)
                random_score = sorted_array[np.array(random_index)]
                query = np.append(query, random_score)
                # query = np.append(query, sorted_array[-groupnum:])
            else:
                random_index = get_random_tobe_queried(index_q, index_q + groupnum, groupnum)
                random_score = sorted_array[np.array(random_index)]
                query = np.append(query, random_score)
                # query = np.append(query, sorted_array[int(index_q):int(index_q + groupnum)])
        index = map(temp_array.index, query)
        return list(index)

    def do_query(self, inputscore, label):
        score = np.copy(inputscore)
        all_score = util.getsum_score(score=score, weight=self.weight)
        if len(all_score) != len(label):
            raise ValueError("len(all_score) != len(label)", len(all_score), len(label))

        index_tobe_queried = self.get_mult_marge_tobe_queried(score=all_score)  # 多样性查询+边界
        for index in index_tobe_queried:
            if index >= len(label):
                raise ValueError("i > len(label)", index, len(label))
            if [index, label[index]] not in self.queried:
                self.queried.append([index, label[index]])
        l = []
        sT = []
        scoreT = np.array(score).T
        for result in self.queried:
            if result[1] == 0:
                l.append(result[1])
            else:
                l.append(1)
            sT.append(scoreT[result[0]])
        s = np.array(sT).T
        return s, l

    ################### Presure test ###############################
    def calculate_test_statistic(self, stresstest, final_stress_index, label, deviation=5):
        __len__ = len(label)

        scores = np.zeros(len(stresstest))
        for index1, stress in enumerate(stresstest):
            for index2, follow in enumerate(stresstest[index1 + 1:]):
                k = (follow - stress) - (follow - stress) // self.seasonal * self.seasonal
                if k <= self.seasonal / 2:
                    if k <= deviation:
                        scores[index1] += 1
                        scores[index1 + index2 + 1] += 1
                else:
                    if self.seasonal - k <= deviation:
                        scores[index1] += 1
                        scores[index1 + index2 + 1] += 1

        for index1, stress in enumerate(stresstest):
            for follow in final_stress_index:
                k = (follow - stress) - (follow - stress) // self.seasonal * self.seasonal
                if k <= self.seasonal / 2:
                    if k <= deviation:
                        scores[index1] += 2
                else:
                    if self.seasonal - k <= deviation:
                        scores[index1] += 2

        return scores
        # max_idx = np.argmax(scores)
        # return test_statistics[max_idx], scores[max_idx]

    def DropSeasonal(self, score, repeat=20, d=0.1, p=0.5):
        '''
        :param score: 异常分数
        :param repeat: 异常按周期出现的次数
        :param deviation: 异常出现周期与规定周期的波动
        :param p: 允许的错差次数
        :return:
        '''
        if d >= 1:
            deviation = d
        else:
            deviation = int(d * self.seasonal)
        if deviation >= self.seasonal / 2:
            raise ValueError("deviation >= self.seasonal/2.", deviation, self.seasonal / 2)
        diff_value_normalize = np.copy(score)  # Utils.norm_min_max(data)
        # threshold = min(0.999, self.truethreshold - 2 / self.seasonal)
        threshold = self.truethreshold
        label = util.score2label_threshold(diff_value_normalize, percentage=threshold)
        anomaly_indices = np.where(label == 1)[0]
        stresstest = []  # 判定为压力测试的点
        __len__ = len(score)
        stressnum = int(__len__ / self.seasonal)  # 有stressnum个压力测试的点
        repeat = min(repeat, stressnum)
        for i in anomaly_indices:
            # 遍历每一个异常
            import math
            if i <= repeat * self.seasonal / 2:
                forward = math.ceil(i / self.seasonal)
                back = repeat - forward
                # forward = 0
            elif i >= __len__ - repeat * self.seasonal / 2:
                back = math.ceil((__len__ - i) / self.seasonal)
                forward = repeat - back
            else:
                back = math.ceil(repeat / 2)
                forward = back

            flag = 0
            for j in range(1, back):
                # 如果在向后找back次数内，都间隔seasonal再次出现异常
                start = max(0, i + self.seasonal * j - deviation)
                end = min(__len__ - 1, i + self.seasonal * j + deviation)
                if sum(label[start:end]) != 0:
                    flag += 1
            for j in range(1, forward):
                # 如果在向前找forward次数内，都间隔seasonal再次出现异常
                start = max(0, i - self.seasonal * j - deviation)
                end = min(__len__ - 1, i - self.seasonal * j + deviation)
                if sum(label[start:end]) != 0:
                    flag += 1
            if flag > p * repeat:
                stresstest.append(i)
        if len(stresstest) == 0:
            label = util.score2label_threshold(diff_value_normalize, percentage=self.prethreshold)
            return label, label, [], score, stresstest
        final_stress_index = []
        for curr in range(len(stresstest)):
            line = (len(final_stress_index) / 2 + len(stresstest)) * p
            stress_score = self.calculate_test_statistic(stresstest, final_stress_index, label,
                                                         deviation)  # 得分最高的标签，及其分数
            if max(stress_score) < line:
                break
            max_idx = int(np.argmax(stress_score))
            final_stress_index.append(stresstest[max_idx])
            stresstest.pop(max_idx)
        if len(final_stress_index) != 0:
            diff_value_normalize[final_stress_index] = 0

        # label = util.score2label_threshold(diff_value_normalize, percentage=self.prethreshold)
        y_stress = np.zeros(len(label))
        y_stress[final_stress_index] = 1

        label[final_stress_index] = 0

        return label, y_stress, diff_value_normalize
