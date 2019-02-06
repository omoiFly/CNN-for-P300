from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from train import get_feature_map, get_feature, train_loaderB, trainA, valA, trainB, valB
from Dataset import Testset
from torch.utils.data import DataLoader
import numpy as np

for i in range(60):
    trainB(i)
    valB(i)

test_loader = DataLoader(
    dataset=Testset('B'),
    shuffle=False,
    batch_size=100
)

clf = svm.SVC()
gnb = GaussianNB()

cols, rows = get_feature(test_loader)
x, y = get_feature_map(train_loaderB)

gnb.fit(x,y)
print(gnb.score(x, y))

clf.fit(x, y)

pred_cols = []
pred_rows = []
for i in range(6):
    pred_cols.append(clf.predict(cols[i,:,:]))
    pred_rows.append(clf.predict(rows[i,:,:]))

pred_rows = np.array(pred_rows)
pred_cols = np.array(pred_cols)

char = [['A', 'B', 'C', 'D', 'E', 'F'],
        ['G', 'H', 'I', 'J', 'K', 'L'],
        ['M', 'N', 'O', 'P', 'Q', 'R'],
        ['S', 'T', 'U', 'V', 'W', 'X'],
        ['Y', 'Z', '1', '2', '3', '4'],
        ['5', '6', '7', '8', '9', '_']]

series = []
real_a = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'
real_b = 'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'

for i in range(100):
    series.append(char[np.argmax(pred_rows[:, i])][np.argmax(pred_cols[:, i])])

series = ''.join(series)
print(series)
counter = 0
for i in range(len(real_a)):
    if real_b[i] == series[i]:
        counter += 1

print(counter / len(real_a))
