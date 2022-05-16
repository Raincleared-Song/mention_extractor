from configs import ConfigBase


class FewNERDMetrics:
    def __init__(self, config: ConfigBase, label_norm=True, ignore_index=-100):
        """
        word_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        """
        self.label_norm = label_norm
        self.ignore_index = ignore_index
        self.y_pred = []
        self.y_true = []
        self.config = config

    @staticmethod
    def __get_class_span_dict__(label, is_string=False):
        """
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        """
        class_span = {}
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span

    @staticmethod
    def __get_intersect_by_entity__(pred_class_span, label_class_span):
        """
        return the count of correct entity
        """
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label, [])))))
        return cnt

    @staticmethod
    def __get_cnt__(label_class_span):
        """
        return the count of entities
        """
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    def metrics_by_entity_(self, pred, label):
        """
        return entity level count of total prediction, true labels, and correct prediction
        """
        pred_class_span = self.__get_class_span_dict__(pred, is_string=True)
        label_class_span = self.__get_class_span_dict__(label, is_string=True)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def get_metrics(self):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        assert len(self.y_pred) == len(self.y_true)
        for i in range(len(self.y_pred)):
            p_cnt, l_cnt, c_cnt = self.metrics_by_entity_(self.y_pred[i], self.y_true[i])
            pred_cnt += p_cnt
            label_cnt += l_cnt
            correct_cnt += c_cnt
        precision = correct_cnt / (pred_cnt + 1e-8)
        recall = correct_cnt / (label_cnt + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {
            "correct_cnt": correct_cnt,
            "predict_cnt": pred_cnt,
            "instance_cnt": label_cnt,
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "micro_f1": round(f1 * 100, 2),
        }

    def expand(self, batch_pred, batch_true):
        y_pred = self.normalize(batch_pred)
        y_true = self.normalize(batch_true)
        self.y_pred += y_pred
        self.y_true += y_true

    def normalize(self, x):
        if not isinstance(x, list):
            x = x.cpu().numpy().tolist()
        assert len(x) > 0 and isinstance(x[0], list)
        assert len(x[0]) > 0 and isinstance(x[0][0], str)
        res = []
        for labs in x:
            cur = []
            for lab in labs:
                if lab in self.config.negative_labels:
                    cur.append("O")
                elif lab.startswith("B-") or lab.startswith("I-"):
                    cur.append(lab[2:])
                else:
                    cur.append(lab)
            res.append(cur)
        return res
