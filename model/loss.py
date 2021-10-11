import torch
import itertools
from .eval_metrics import SI_SDR

def uPIT(est_s, s, metrics=SI_SDR):
    if s.shape[2] != est_s.shape[2]:
        raise ValueError("C does not match with the number of speakers")
    batch_size, _ ,C = s.shape
    permutations = list(itertools.permutations(range(C)))

    # calc loss minimize permutation for each batch
    min_permutations = [0]*batch_size
    for b in range(batch_size):
        with torch.no_grad():
            min_loss = torch.tensor(float('inf'))
            for p in permutations:
                loss = 0
                for i in range(C):
                    loss -= metrics(est_s[b,:,i], s[b,:,p[i]]) / C

                if loss < min_loss:
                    min_loss = loss
                    min_permutations[b] = (p)

    # calc total loss
    total_loss = 0
    for b in range(batch_size):
        uPIT_loss = 0
        perm = min_permutations[b]
        for i in range(C):
            uPIT_loss -= metrics(est_s[b,:,i], s[b,:,perm[i]]) / C
        total_loss += uPIT_loss

    total_loss = total_loss/batch_size

    return total_loss


if __name__ == "__main__":
    s1 = torch.randn(1000).unsqueeze(1).unsqueeze(0)
    s2 = torch.randn(1000).unsqueeze(1).unsqueeze(0)
    s3 = torch.randn(1000).unsqueeze(1).unsqueeze(0)
    s = torch.cat([s1, s2, s3],dim=2)

    n = 0.1*torch.randn(1000).unsqueeze(1).unsqueeze(0)

    est_s1 = s2+n
    est_s2 = s3+n
    est_s3 = s1+n
    est_s = torch.cat([est_s1, est_s2, est_s3], dim=2)

    # print(s.shape, est_s.shape)
    print(uPIT(est_s,s))