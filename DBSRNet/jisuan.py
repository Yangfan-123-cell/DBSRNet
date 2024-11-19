import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy import stats

pred = []
score = []
with open ("checkpoints/ahiq_pipal/score.txt","r") as listFile:
    for line in listFile:
        a,b,c =line[:-1].split(",")
        pred.append(b)
        score.append(c)

pred = np.array(pred)
score = np.array(score)
pred = pred.astype(float)
score = score.astype(float)

rho_s, _ = spearmanr(np.squeeze(pred), np.squeeze(score))
rho_p, _ = pearsonr(np.squeeze(pred), np.squeeze(score))
rmse = np.sqrt(((np.squeeze(pred)-np.squeeze(score))**2).mean())
rho_k, _ = stats.kendalltau(np.squeeze(pred), np.squeeze(score))

print('SRCC:{:.4} / PLCC:{:.4} / RMSE:{:.4} / KROCC:{:.4}'.format(rho_s, rho_p,rmse,rho_k))