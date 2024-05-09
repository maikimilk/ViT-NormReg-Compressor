import pruning
import timm
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from tqdm import tqdm
from thop import profile
import random
import numpy as np


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def main():
    torch_fix_seed(777)

    torch.autograd.set_detect_anomaly(True)

    # tk, tk_n, evit, evit_n, n_wa, a_wn の手法を選択
    model = "tk"
    # バッチサイズを指定
    batch = "32"
    # データセットの指定
    data_name = "CIFAR10"

    #データセット
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size = (224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops = 3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    if data_name == "STL10":
        train_set = torchvision.datasets.STL10(root = './data', split = 'train',
                                            download=True, transform=train_transform)

        test_set = torchvision.datasets.STL10(root = './data', split = 'test',
                                            download=True, transform=test_transform)
    elif data_name == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transform)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=test_transform)

    #　バッチサイズ
    batch_size = int(batch)

    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = data_loader.dataset.classes

    #モデル読み込み
    net = timm.create_model('vit_small_patch16_224_in21k', pretrained=True, num_classes=10)


    #モデルの調整
    pruning.patch.apply_patch(net, select_method = model, pool_k = 15)


    #エポック数
    num_epochs = 100

    # エポックごとに訓練とテストの正解率を保存するリスト
    train_accuracies = []
    test_accuracies = []

    #損失関数の定義
    criterion = nn.CrossEntropyLoss()

    #最適化アルゴリズム
    optimizer = optim.AdamW(net.parameters(), lr=5e-4 * batch_size/512., weight_decay = 1e-4)

    vis_iter = 5

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    #FLOPs計測
    sample = torch.randn(1,1,3,224,224).to(device)
    net.eval()

    flops, parameters = profile(net, sample)
    print(f'FLOPs: {flops/1e+9:.2f} G')
    print(f'Prams: {parameters/1e+6:.2f} M')


    torch_fix_seed(777)

    best_acc = 0

    for epoch in range(num_epochs):
        train_correct = 0
        test_correct = 0
        train_loss = 0
        test_loss = 0

        net.train()
        for i, (x, label) in tqdm(enumerate(data_loader)):
            x, label = x.to(device), label.to(device)
            # 画像を vit へ入力し、出力を取得
            y = net(x)
            # 損失の計算
            loss = criterion(y, label)
            # 一つ前の更新値を初期化
            optimizer.zero_grad()
            # バックプロパゲーション
            loss.backward()
            # 重み更新
            optimizer.step()

            # 予測値が正解ラベルと一致したらカウント
            pred = y.argmax(-1)
            train_correct = train_correct + (pred == label).sum().item()
            train_loss = train_loss + loss.item()

        # テストデータで推論
        net.eval()
        for i, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            y = net(x)
            loss = criterion(y, label)
            test_loss = test_loss + loss.item()
            pred = y.argmax(-1)
            test_correct = test_correct + (pred == label).sum().item()

        # 各エポックでの訓練とテストの正解率を計算
        train_accuracy = (train_correct / len(data_loader.dataset)) * 100
        test_accuracy = (test_correct / len(test_loader.dataset)) * 100

        # 正解率をリストに保存
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        #　エポックごとの結果を表示
        if (epoch+1)%1 == 0:
            mem_result_epock = 'Epoch [{}/{}]\n'.format(epoch+1, num_epochs) +\
                'Train\tLoss : {:.3f}\t'.format(train_loss/len(data_loader)) +\
                'Acc : {:.2f}\n'.format((train_correct/len(data_loader.dataset))*100) +\
                'Test\tLoss : {:.3f}\t'.format(test_loss/len(test_loader)) + \
                'Acc : {:.2f}\n'.format((test_correct/len(test_loader.dataset))*100) + '\n'

            print(mem_result_epock)

    # リストをテンソルに変換
    test_accuracies_tensor = torch.tensor(test_accuracies)
    # 最大値を見つける
    max_value = torch.max(test_accuracies_tensor)
    mem_max_acc = '最大値 : {:.2f}\n'.format(max_value.item())
    print(mem_max_acc)

if __name__ == '__main__':
    main()
