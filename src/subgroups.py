import numpy as np

THRES = 0.5
GROUP_THRES = 0


class SubgroupDetectorBase:
    subg_treatments = None

    @staticmethod
    def detect(x, pred_y_a01):
        raise NotImplementedError()

    @staticmethod
    def detect_with_a(x, a, pred_y_a):
        raise NotImplementedError()


class SubgroupDetectorSimple(SubgroupDetectorBase):
    subg_treatments = np.array([0, 1])

    @staticmethod
    def detect(x, pred_y_a01):
        return pred_y_a01 < THRES

    @staticmethod
    def detect_with_a(x, a, pred_y_a):
        pred_class = pred_y_a < THRES
        return np.concatenate(
            [
                pred_class * (a == 0),
                pred_class * (a == 1),
            ],
            axis=1,
        )


class SubgroupDetector(SubgroupDetectorBase):
    subg_treatments = np.array([0, 1, 0, 0, 1, 1])

    @staticmethod
    def _get_subgroup(X):
        return ((X[:, :1] > -1) & (X[:, :1] < 2)) & (np.abs(X[:, 1:2]) < 2.5)

    @staticmethod
    def detect(x, pred_y_a01):
        pred_class_a01 = pred_y_a01 < THRES
        subg_mask = SubgroupDetector._get_subgroup(x)
        not_subg_mask = np.logical_not(subg_mask)
        return np.concatenate(
            [
                pred_class_a01[:, :1],
                pred_class_a01[:, 1:],
                subg_mask * pred_class_a01[:, :1],
                not_subg_mask * pred_class_a01[:, :1],
                subg_mask * pred_class_a01[:, 1:],
                not_subg_mask * pred_class_a01[:, 1:],
            ],
            axis=1,
        )

    @staticmethod
    def detect_with_a(x, a, pred_y_a):
        pred_class = pred_y_a < THRES
        subg_mask = SubgroupDetector._get_subgroup(x)
        not_subg_mask = np.logical_not(subg_mask)
        return np.concatenate(
            [
                pred_class * (a == 0),
                pred_class * (a == 1),
                subg_mask * pred_class * (a == 0),
                not_subg_mask * pred_class * (a == 0),
                subg_mask * pred_class * (a == 1),
                not_subg_mask * pred_class * (a == 1),
            ],
            axis=1,
        )


class ScoreSubgroupDetector(SubgroupDetectorBase):
    subg_treatments = np.array([0, 1, 0, 0, 1, 1])

    @staticmethod
    def detect(x, a, pred_y_a):
        subg_mask = SubgroupDetector._get_subgroup(x)
        not_subg_mask = np.logical_not(subg_mask)
        return np.concatenate(
            [
                (a == 0),
                (a == 1),
                subg_mask * (a == 0),
                not_subg_mask * (a == 0),
                subg_mask * (a == 1),
                not_subg_mask * (a == 1),
            ],
            axis=1,
        )
