import torch
import torch.nn as nn
import torch.nn.functional as F


class Vanilla(nn.Module):
    def __init__(self):
        super(Vanilla, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(64, 5, 5, padding=2),
            nn.MaxPool1d(2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(5*120, 30),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(30, 2),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.conv(x)
        fc = self.fc(x.view(x.size(0), -1))
        x = self.out(fc).squeeze()

        return x, fc


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(     # LSTM 效果要比 nn.RNN() 好多了
            input_size=240,      # 图片每行的数据像素点
            hidden_size=64,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 2)    # 输出层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out, 0


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Conv1d(in_channels, out_channels, 3, padding=1)

        self.conv1 = nn.Conv1d(out_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        x = self.conv0(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv1d(64, self.in_channels, 3)
        self.bn = nn.BatchNorm1d(self.in_channels)

        self.resblk = ResBlock(self.in_channels, 6)
        self.sameblk = ResBlock(6, 6)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(6*238, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.resblk(x)
        x = self.dropout(x)
        out = self.fc(x.view(x.size(0), -1))

        return out, x.view(x.size(0), -1)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv_down = nn.Sequential(
            nn.Conv1d(64, 5, 5, padding=2),
            nn.MaxPool1d(2),
            nn.ReLU(),
        )

        self.dense_down = nn.Sequential(
            nn.Linear(5*120, 30),
            nn.ReLU(),
        )

        self.dense_up = nn.Sequential(
            nn.Linear(30, 5*120),
        )

        # 需要 reshape 回 128 * 58
        self.conv_up = nn.Sequential(
            nn.ConvTranspose1d(5, 64, 5, padding=2),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x):
        x = self.conv_down(x)
        hidden = self.dense_down(x.view(-1, 5*120))
        out = self.dense_up(hidden)
        out = self.conv_up(out.view(-1, 5, 120))

        return out, hidden


class InstructedAE(nn.Module):
    def __init__(self):
        super(InstructedAE, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 10, 5, padding=2),
            nn.MaxPool1d(4),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(10*60, 50),
            nn.ReLU(),
        )

        self.linear_upsample = nn.Sequential(
            nn.Linear(50, 10*60),
            nn.ReLU(),
        )

        self.conv_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv1d(10, 64, 5, padding=2),
        )

    def forward(self, x):
        x = self.conv1(x).view(x.size(0), -1)
        hidden = self.fc1(x)
        out = self.linear_upsample(hidden)
        out = self.conv_upsample(out.view(out.size(0), 10, -1))

        return out, hidden


if __name__ == '__main__':
    import torch.onnx as onnx
    model = Vanilla()
    model.train(False)
    onnx.export(model, torch.zeros([85, 64, 240]), './vanilla_cnn')