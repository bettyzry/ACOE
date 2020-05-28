import util
import random
import numpy as np
from BaseDetectors.getScore import GetScore


class StreamACOE:
    def __init__(self, weight, winsize=100000, threshold=0.99, query_rate=0.02, seasonal=1440, batch=64,
                 history=None, feedback=None, withstress=True):
        self.weight = weight
        self.winsize = winsize
        self.threshold = threshold
        self.query_rate = query_rate
        self.seasonal = seasonal
        self.batch = batch
        self.withstress = withstress

        self.history = np.array(history)  # value
        self.feedback = np.array(feedback)  # 1 for outlier, 0 for inlier, 2 for stresstest, -1 for unknown
        self.score = np.zeros(len(history))  # [[ma,ewma,holt,zscore,sesd]]
        return

    def fit(self, newdata, feedback):
        ts = None
        self.history = np.concatenate([self.history, newdata])
        self.feedback = np.concatenate([self.feedback, feedback])
        if len(self.history) > self.winsize:
            self.history = self.history[-self.winsize:]
            self.feedback = self.feedback[-self.winsize:]
        self.score = GetScore(self.history, self.threshold, self.seasonal)
        if self.withstress:
            is_anomaly, need_labeled, is_stresstest = self.withstress_query(ts, self.score)
        else:
            is_anomaly, need_labeled, is_stresstest = self.dropstress_query(ts, self.score)
        return is_anomaly, need_labeled, is_stresstest

    def update_feedback(self, feedback):
        self.feedback = np.concatenate([self.feedback[:-self.batch], feedback])

    def dropstress_query(self, ts, score_list):
        # 获取集成后的异常分数 score
        score_vec = np.copy(score_list)
        score = np.dot(self.weight, score_vec)

        # 判断是否为异常，是否需要查询
        anomaly = util.score2label_threshold(self.threshold)
        is_anomaly = anomaly[-self.batch:]
        stresstest = self.getstresstest(score)
        is_stresstest = stresstest[-self.batch:]
        need_labeled = self.needlabeled(score)

        return is_anomaly, need_labeled, is_stresstest

    def withstress_query(self, ts, score_list):
        # 获取集成后的异常分数 score
        score_vec = np.copy(score_list)
        score = np.dot(self.weight, score_vec)

        # 判断是否为异常，是否需要查询
        anomaly = util.score2label_threshold(score, self.threshold)
        is_anomaly = anomaly[-self.batch:]
        need_labeled = self.needlabeled(score)

        return is_anomaly, need_labeled, 0

    def calculate_test_statistic(self, stresstest, final_stress_index, deviation=5):
        '''
        :param stresstest: 初步判定为压力测试点的标签
        :param final_stress_index: 已确定为压力测试点的标签
        :param feedback: DataFrame([[2019, 1, 2, 3, 4, 5, 0]], columns=['timestamp', 'ewma', 'ma', 'holt_winters', 'zscore', 'sesd', 'anomaly_type'])
        :param deviation: 异常出现周期与规定周期的波动
        :return:
        '''
        scores = np.zeros(len(stresstest))
        # 在初步确定的点内索引
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

        # 在最终确定的点内索引
        for index1, stress in enumerate(stresstest):
            for follow in final_stress_index:
                k = (follow - stress) - (follow - stress) // self.seasonal * self.seasonal
                if k <= self.seasonal / 2:
                    if k <= deviation:
                        scores[index1] += 2
                else:
                    if self.seasonal - k <= deviation:
                        scores[index1] += 2

        queried_stress = np.where(self.feedback == 2)[0]

        # 在查询所得确定为压力测试的点内索引
        for index1, stress in enumerate(stresstest):
            for follow in queried_stress:
                k = (follow - stress) - (follow - stress) // self.seasonal * self.seasonal
                if k <= self.seasonal / 2:
                    if k <= deviation:
                        scores[index1] += 4
                else:
                    if self.seasonal - k <= deviation:
                        scores[index1] += 4
        return scores

    def getstresstest(self, score, repeat=20, d=0.1, p=0.5):
        '''
        :param score: 异常分数
        :param self.feedback: 1:outlier, 0:inlier, -1:unlabeled, 2:presure test
        :param repeat: 异常按周期出现的次数
        :param d: 异常出现周期与规定周期的波动，占比
        :param p: 允许的错差次数，占比
        :return: 判定为压力测试的点的标签
        '''
        if d >= 1:
            deviation = d
        else:
            deviation = int(d * self.seasonal)
        if deviation >= self.seasonal / 2:
            raise ValueError("deviation >= self.seasonal/2.", deviation, self.seasonal / 2)
        diff_value_normalize = np.copy(score)  # util.norm_min_max(data)
        threshold = min(0.999, self.threshold - 2 / self.seasonal)
        label = util.score2label_threshold(diff_value_normalize, percentage=threshold)

        anomaly_indices = np.where(label == 1)[0]
        stresstest = []  # 判定为压力测试的点
        __len__ = len(score)
        stressnum = __len__ // self.seasonal  # 有stressnum个压力测试的点
        repeat = min(repeat, stressnum)
        for i in anomaly_indices:
            # 遍历每一个异常
            if i <= repeat * self.seasonal / 2:
                forward = i // self.seasonal
                back = repeat - forward
            elif i >= __len__ - repeat * self.seasonal / 2:
                back = (__len__ - i) // self.seasonal
                forward = repeat - back
            else:
                back = repeat // 2
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
            label_stress = np.zeros(len(score))
            return label_stress

        final_stress_index = []
        for curr in range(len(stresstest)):
            line = (len(final_stress_index) / 2 + len(stresstest)) * p
            stress_score = self.calculate_test_statistic(stresstest, final_stress_index,
                                                         deviation)  # 得分最高的标签，及其分数
            if max(stress_score) < line:
                break
            max_idx = int(np.argmax(stress_score))
            final_stress_index.append(stresstest[max_idx])
            stresstest.pop(max_idx)
        if len(final_stress_index) != 0:
            diff_value_normalize[final_stress_index] = 0

        label_stress = np.zeros(len(score))
        label_stress[final_stress_index] = 1
        return label_stress

    def needlabeled(self, score):
        # 获取查询点中的异常占比
        feedback = np.copy(self.feedback)
        feedback[feedback == 2] = 1

        index = np.where(feedback != -1)[0]
        label = np.array(self.feedback)[index]

        anomaly_rate = sum(label) / len(label) if sum(label) != 0 else 0.000001


        # 获取新输入点的类型 informativeness 1, representativeness 0
        # 计算查询概率
        representativeness = self.threshold ** (self.threshold / anomaly_rate)
        informativeness = 1 - representativeness



        need_labeled_list = []
        down = self.get_line(score, max(self.threshold - 0.05, 0))
        up = self.get_line(score, min(self.threshold + 0.05, 1))
        group = np.zeros(self.batch)
        for i in range(self.batch):
            newscore = score[i - self.batch]
            if newscore <= up and newscore >= down:
                group[i] = 1 # informativeness
            else:
                group[i] = 0 # representativeness
        num_inf = sum(group)/self.batch
        if num_inf == 0:
            representativeness = representativeness * self.query_rate / (1 - num_inf)
            informativeness = 0
        elif num_inf == 1:
            representativeness = 0
            informativeness = informativeness * self.query_rate / num_inf
        else:
            representativeness = representativeness * self.query_rate / (1-num_inf)
            informativeness = informativeness * self.query_rate/num_inf

        print(representativeness, informativeness, anomaly_rate)
        for i in group:
            # 以rate的概率进行查询
            alpha = random.uniform(0, 1)
            if i == 1:
                rate = informativeness
            else:
                rate = representativeness
            if alpha <= rate:
                need_labeled = 1
            else:
                need_labeled = 0
            need_labeled_list.append(need_labeled)
        return need_labeled_list

    def update_weight(self):
        # 获取偏置w
        feedback = np.copy(self.feedback)
        feedback[feedback == 2] = 1

        index = np.where(feedback!=-1)[0]

        # score = np.copy(self.score)    # [[MA],[EWMA],[Holt],[Zscore],[SESD]]


        score = [self.score[i][index] for i in range(len(self.score))]
        label = np.array(self.feedback)[index]

        from sklearn.metrics import accuracy_score
        w = np.zeros(len(score))
        for i in range(len(score)):
            singlescore = score[i]
            # singlescore = singlescore[index]
            pre_label = util.score2label_threshold(singlescore, self.threshold)
            f = accuracy_score(label, pre_label)
            w[i] = f

        # 通过生成200个随机权重来确定新的权重weight
        loss = [10000]
        count = 1
        last_loss = 0
        while abs(last_loss - min(loss)) / len(label) > 0.001:
            last_loss = min(loss)
            weightlist = [self.weight]
            for i in range(200):
                # 产生随机权重
                weight = np.zeros(len(w))
                for j in range(len(w)):
                    alpha = random.uniform(1, 1 + 1 / count)
                    mul_div = random.random()
                    if mul_div <= w[j]:
                        weight[j] = self.weight[j] * alpha
                    else:
                        weight[j] = self.weight[j] / alpha
                weight = util.norm_to_1(weight)
                weightlist.append(weight)
            loss = []
            for weight in weightlist:
                l1 = self.loss(score, label, weight)
                loss.append(l1)
            index = loss.index(min(loss))
            partbest_weight = weightlist[index]
            self.weight = partbest_weight
            count += 1
        return

    def loss(self, score, label, weight):
        # 将权重和异常分数加权
        all_score = np.dot(weight, score)

        # 计算loss
        from pandas import Series
        data = Series(all_score)
        rank_data = data.sort_values(ascending=True)
        index = rank_data.index
        sum = 0
        for i in range(len(index)):
            if label[index[i]] == 1:
                sum += len(index) - i  # 1 - i/len(index)
            else:
                sum += i
        l1 = sum / len(index)
        return l1

    def get_line(self, score, percentage):
        import math
        temp_array = score.copy()
        temp_array.sort()
        line_index = len(temp_array) - 1 if percentage == 1 else math.floor(len(temp_array) * percentage)
        line = temp_array[line_index]
        return line
