""" Eval metrics and related
Hacked together by / Copyright 2020 Ross Wightman
Source: https://github.com/rwightman/pytorch-image-models/blob/e7da205345dcf770ee4bedd62d06fad7a1458904/timm/utils/metrics.py
"""


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size
        for k in topk
    ]


def test_gcp_connection():
    """Tests connection to GCP based on the presense of an environment variable.

    Raises:
        RuntimeError: If connection can't be made, it will raise a RuntimeError.
    """
    from google.cloud import storage

    try:
        storage.Client()
        print("[INFO] GCP connection successful! Data/models will be saved to GCP.")
    except:
        raise RuntimeError(
            "GCP connection unsuccessful, this is required for storing data and models, check GOOGLE_APPLICATION_CREDENTIALS"
        )
