import torch
import itertools
from eval_utils import SI_SDR

def uPIT(est_s, s, eval_idx = "sisdr"):
    if s.shape[2] != est_s.shape[2]:
        raise ValueError("C does not match with the number of speakers")
    batch_size, _ ,C = s.shape
    permutations = list(itertools.permutations(range(C)))

    total_loss = 0

    for b in range(batch_size):
        min_loss = torch.tensor(float('inf'))
        for p in permutations:
            eval_value = []
            for i in range(C):
                if eval_idx=="sisdr":
                    eval_value.append(SI_SDR(est_s[b,:,i], s[b,:,p[i]]))
            loss = -sum(eval_value)/C
            if loss < min_loss:
                min_loss = loss
                min_eval_value = eval_value
        total_loss += min_loss

    total_loss = total_loss/batch_size

    return total_loss


if __name__ == "__main__":
    s1 = torch.rand(1000)
    s2 = torch.rand(1000)
    s3 = torch.rand(1000)
    list_s = [s1, s2, s3]

    n = 0.1*torch.rand(1000)

    est_s1 = s2+n
    est_s2 = s3+n
    est_s3 = s1+n
    list_est_s = [est_s1, est_s2, est_s3]

    print(PIT(list_est_s,list_s))