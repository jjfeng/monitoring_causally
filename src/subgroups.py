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
    def detect(x):
        return np.ones((x.shape[0], 1))

    @staticmethod
    def detect_with_a(x, a):
        return np.concatenate(
            [
                (a == 0),
                (a == 1),
            ],
            axis=1,
        )


class SubgroupDetector(SubgroupDetectorBase):
    subg_treatments = np.array([0, 1, 0, 0, 1, 1])

    @staticmethod
    def _get_subgroup(X):
        # x0 is -1 to 2, x1 is -2.5 to 2.5
        # more people in this group will be treated with A=0
        return ((X[:, :1] > -1) & (X[:, :1] < 2)) & (np.abs(X[:, 1:2]) < 2.5)

    @staticmethod
    def detect(x:np.ndarray):
        subg_mask = SubgroupDetector._get_subgroup(x)
        not_subg_mask = np.logical_not(subg_mask)
        return np.concatenate(
            [
                np.ones((x.shape[0],1)),
                np.ones((x.shape[0],1)),
                subg_mask,
                not_subg_mask,
                subg_mask,
                not_subg_mask,
            ],
            axis=1,
        )

    @staticmethod
    def detect_with_a(x, a):
        h = SubgroupDetector.detect(x)
        a_mask = a == SubgroupDetector.subg_treatments
        return h * a_mask


class ScoreSubgroupDetector(SubgroupDetectorBase):
    subg_treatments = np.array([0, 1, 0, 0, 1, 1])

    @staticmethod
    def detect(x, a):
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
