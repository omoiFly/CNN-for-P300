import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import hiddenlayer as hl
import torch.onnx as onnx
from Dataset import Trainset, Testset
from train_valid_split import train_valid_split
from Model import Vanilla, AutoEncoder, InstructedAE, ResCNN, RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Vanilla().to(device)

random_seed = np.random.seed(666)

trainsetA, valsetA = train_valid_split(Trainset('A'), random_seed=random_seed)
trainsetB, valsetB = train_valid_split(Trainset('B'), random_seed=random_seed)

train_loaderA = DataLoader(
    dataset=trainsetA,
    batch_size=85,
    shuffle=True,
)

valid_loaderA = DataLoader(
    dataset=valsetA,
    batch_size=85
)

train_loaderB = DataLoader(
    dataset=trainsetB,
    batch_size=85,
    shuffle=True,
)

valid_loaderB = DataLoader(
    dataset=valsetB,
    batch_size=85
)

test_loaderA = DataLoader(
    dataset=Testset('A'),
    shuffle=False
)

test_loaderB = DataLoader(
    dataset=Testset('B'),
    shuffle=False
)

mse_criterion = nn.MSELoss()
cross_entropy_criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD([
#     {'params':model.conv1.parameters()},
#     {'params':model.conv2.parameters()},
#     {'params':model.fc.parameters()},
#     {'params':model.out.parameters(), 'lr': 1e-8}], lr=5e-4, momentum=0.9, weight_decay=1e-4)
#
# ae_optimizer = optim.Adam([
#     {'params':model.linear_upsample.parameters()},
#     {'params':model.conv_upsample.parameters()}], lr=1e-3, weight_decay=1e-4)

optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9,
                      weight_decay=1e-4)

if False:
    state = torch.load('./ae')
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])


def trainA(ep):
    model.train()
    total_loss = 0
    total_acc = 0
    for step, (x, y) in enumerate(train_loaderA):
        data, target = x.to(device), y.to(device).long()
        output, _ = model(data)

        optimizer.zero_grad()
        loss = cross_entropy_criterion(output, target)
        total_loss += loss
        loss.backward()

        optimizer.step()

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).sum()
        total_acc += acc.item() / train_loaderA.batch_size

    avg_loss = total_loss / len(train_loaderA)
    avg_acc = total_acc / len(train_loaderA)
    print("Epoch: {} Loss: {},  Acc: {}".format(ep, avg_loss,
                                                     avg_acc))

    if ep == 499:
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'acc': total_acc
        }, './vallina_cnn_A')


def trainB(ep):
    model.train()
    total_loss = 0
    total_acc = 0
    for step, (x, y) in enumerate(train_loaderB):
        data, target = x.to(device), y.to(device).long()
        output, _ = model(data)

        optimizer.zero_grad()
        loss = cross_entropy_criterion(output, target)
        total_loss += loss
        loss.backward()

        optimizer.step()

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).sum()
        total_acc += acc.item() / train_loaderB.batch_size

    avg_loss = total_loss / len(train_loaderB)
    avg_acc = total_acc / len(train_loaderB)
    print("Epoch: {} Loss: {},  Acc: {}".format(ep, avg_loss,
                                                     avg_acc))

    if ep == 499:
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'acc': total_acc
        }, './vallina_cnn_B')



def valA(ep):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for step, (x, y) in enumerate(valid_loaderA):
            data, target = x.to(device), y.to(device).long()
            output, _ = model(data)

            loss = cross_entropy_criterion(output, target)
            total_loss += loss

            predict = output.data.max(1)[1]
            acc = predict.eq(target.data).sum()
            total_acc += acc.item() / valid_loaderA.batch_size
    print("Valid Epoch: {} Loss: {},  Acc: {}".format(ep, total_loss / len(valid_loaderA),
                                                           total_acc / len(valid_loaderA)))


def valB(ep):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for step, (x, y) in enumerate(valid_loaderB):
            data, target = x.to(device), y.to(device).long()
            output, _ = model(data)

            loss = cross_entropy_criterion(output, target)
            total_loss += loss

            predict = output.data.max(1)[1]
            acc = predict.eq(target.data).sum()
            total_acc += acc.item() / valid_loaderB.batch_size
    print("Valid Epoch: {} Loss: {},  Acc: {}".format(ep, total_loss / len(valid_loaderB),
                                                           total_acc / len(valid_loaderB)))


def predictA():
    char = [['A', 'B', 'C', 'D', 'E', 'F'],
            ['G', 'H', 'I', 'J', 'K', 'L'],
            ['M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X'],
            ['Y', 'Z', '1', '2', '3', '4'],
            ['5', '6', '7', '8', '9', '_']]

    series = []
    real_a = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'
    model.eval()
    with torch.no_grad():
        for step, (cols, rows) in enumerate(test_loaderA):
            col_pred_set = []
            row_pred_set = []
            # print("     0 - 6")
            for line in range(6):
                data = cols[:, line, :, :].to(device)
                output, _ = model(data)
                col_pred_set.append(output.data.cpu().numpy())
            # print("     7 - 12")
            for line in range(6):
                data = rows[:, line, :, :].to(device)
                output, _ = model(data)
                row_pred_set.append(output.data.cpu().numpy())
            col_pred_set = np.array(col_pred_set).squeeze()
            row_pred_set = np.array(row_pred_set).squeeze()
            col_pred = np.argmax(col_pred_set, axis=0)[1]
            row_pred = np.argmax(row_pred_set, axis=0)[1]

            series.append(char[row_pred][col_pred])
    series = ''.join(series)
    print(series)
    counter = 0
    for i in range(len(real_a)):
        if real_a[i] == series[i]:
            counter += 1
    print(counter / len(real_a))
    return counter / len(real_a)


def predictB():
    char = [['A', 'B', 'C', 'D', 'E', 'F'],
            ['G', 'H', 'I', 'J', 'K', 'L'],
            ['M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X'],
            ['Y', 'Z', '1', '2', '3', '4'],
            ['5', '6', '7', '8', '9', '_']]

    series = []
    real_b = 'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'
    model.eval()
    with torch.no_grad():
        for step, (cols, rows) in enumerate(test_loaderB):
            col_pred_set = []
            row_pred_set = []
            # print("     0 - 6")
            for line in range(6):
                data = cols[:, line, :, :].to(device)
                output, _ = model(data)
                col_pred_set.append(output.data.cpu().numpy())
            # print("     7 - 12")
            for line in range(6):
                data = rows[:, line, :, :].to(device)
                output, _ = model(data)
                row_pred_set.append(output.data.cpu().numpy())
            col_pred_set = np.array(col_pred_set).squeeze()
            row_pred_set = np.array(row_pred_set).squeeze()
            col_pred = np.argmax(col_pred_set, axis=0)[1]
            row_pred = np.argmax(row_pred_set, axis=0)[1]

            series.append(char[row_pred][col_pred])
    series = ''.join(series)
    print(series)
    counter = 0
    for i in range(len(real_b)):
        if real_b[i] == series[i]:
            counter += 1
    print(counter / len(real_b))
    return counter / len(real_b)

def get_feature_map(loader, contain_label=True):
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            _, feature = model(data)
        print(feature.shape)
        return feature.cpu().numpy(), target.cpu().numpy()


def get_feature(loader):
    with torch.no_grad():
        col_pred_set = []
        row_pred_set = []
        for step, (cols, rows) in enumerate(loader):
            # print("     0 - 6")
            for line in range(6):
                data = cols[:, line, :, :].to(device)
                _, output = model(data)
                col_pred_set.append(output.data.cpu().numpy())
            # print("     7 - 12")
            for line in range(6):
                data = rows[:, line, :, :].to(device)
                _, output = model(data)
                row_pred_set.append(output.data.cpu().numpy())

        return np.array(col_pred_set), np.array(row_pred_set)


def train_AE(ep):
    model.train()
    total_loss = 0
    for step, (x, y) in enumerate(train_loaderB):
        data = x.to(device)
        output, _ = model(data)

        optimizer.zero_grad()
        loss = mse_criterion(output, data)
        total_loss += loss
        loss.backward()

        optimizer.step()

    avg_loss = total_loss / len(train_loaderB)
    print("Epoch: {} Loss: {}".format(ep, avg_loss))

    if ep % 500 == 0 and ep > 0:
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, './ae')


if __name__ == '__main__':
    best_loss = 30
    i = 0
    for i in range(150):
        trainA(i)
        valA(i)
    predictA()

#    onnx.export(model,
#                torch.zeros([85, 64, 240]).to(device),
#                'vallina_cnn.onnx',
#                verbose=True)
        # if loss < best_loss:
        #     best_loss = loss
        #     no_improve_counter = 0
        # no_improve_counter += 1
        #
        # if no_improve_counter > 200:
        #     torch.save({
        #         'epoch': i,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': best_loss
        #     }, './autoencoder_val')
        #     print('early stop with 200 no improvement')
        #     break
        #
        #
