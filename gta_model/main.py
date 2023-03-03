import os.path
import torch.nn as nn
import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from customdata import my_customdata
import pandas as pd
import sys
import copy

from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

device = torch.device("cuda")
#pip install torchsummary를 통해서모델의 크기, 파라메터 등등을 알 수 있음
#pip install pytorch-model-summary 이런걸로 볼 수도 있음
#pip install torchsummaryX 이게 최신임
#pip install torchinfo




#데이터 어규먼트 부분
def aug_function(mode_flag):
    if mode_flag =="train":
        train_transform = A.Compose([
            # A.SmallestMaxSize(max_size=160),
            A.Resize(width=224, height=224),
            # A.ShiftScaleRotate(shift_limit=0.05,scale_limit=0.06, rotate_limit=20, p=0.7),
            # A.RGBShift(r_shift_limit=17,g_shift_limit=17,b_shift_limit=17,p=0.7),
            # A.RandomBrightnessContrast(p=1),
            # A.RandomShadow(p=0.7),
            # A.RandomFog(p=0.7),
            # #A.RandomSnow(p=0.6), 눈넣으니까 이미지 개판됨;
            # A.HorizontalFlip(p=0.7),
            # A.VerticalFlip(p=0.7),
            # A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
        return train_transform

        if mode_flag == "test":
            train_transform = A.Compose([
                # A.SmallestMaxSize(max_size=160),
                A.Resize(width=224, height=224),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()

            ])
            return test_transform


        #트레인은 랜덤, 고정 둘다 넣어도됨
    elif mode_flag =="val":
        val_transform = A.Compose([
            # A.SmallestMaxSize(max_size=160),
            A.Resize(width=224, height=224),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()

            #여긴 랜덤을 넣으면안됨
        ])
        return val_transform


#어그먼트 플래그 설정
train_aug = aug_function(mode_flag="train")
# test_aug = aug_function(mode_flag="test")
val_aug = aug_function(mode_flag="val")
#데이터 로더 부분
train_dataset = my_customdata("./dataset/train/", transform=train_aug)
#test_dataset = my_customdata("./dataset/test/", transform=test_aug)
val_dataset = my_customdata("./dataset/valid/",transform=val_aug)

# for i in test_dataset:
#     print(i)
#     exit()

#어규먼트 이미지 확인
def visulize_agumentations(dataset, idx=0, samples=20,cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(

        t ,(A.Normalize, ToTensorV2)
    )])
    rows = samples//cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize =(12,6))
    for i in range(samples):
        image,_ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
#visulize_agumentations(train_dataset)

def main() :
    #데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #pin_memory=True)
                              #num_workers=2
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

                            #pin_memory=True)
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                             #pin_memory=True)


    #모델 모르겟으면
    #print(net)써서 직접보고나서 마지막 피쳐와 아웃풋을 맞춰주면됨
    #아웃풋은 나갈 라벨의 갯수  인 피쳐는 마지막 인피쳐에 맞춰주면됨

    #swin t 모델
    # net = models.swin_t(weights='IMAGENET1K_V1')
    # net.head = nn.Linear(in_features=768, out_features=450)
    # net.to(device)

    #resnet50 모델
    net = models.resnet50(pretrained=True)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=7, padding=3, bias=False)
    net.fc = nn.Linear(in_features=2048,out_features=5)#450개로 분류하잖음
    net.to(device)


    # net = models.resnet18(pretrained=True)
    # #num_ftrs = net.fc.in_features
    # net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=7, padding=3, bias=False)
    # net.fc = nn.Linear(in_features=512,out_features=5)#450개로 분류하잖음
    # net.to(device)


    #resnet18 모델
    # net = models.resnet18(pretrained=True)
    # net.conv1 = nn.Linear(net.conv1.in_features, 64*3*7*7)
    # net.fc = nn.Linear(in_features=516,out_features=5)#450개로 분류하잖음
    # net.to(device)

    #
    # device = torch.device("cuda")  ##Assigning the Device which will do the calculation
    # net = torch.load("bestresnet50.pt")  # Load model to CPU
    # net = net.to(device)  # set where to run the model and matrix calculation
    # net.eval()  # set the device to eval() mode for testing

    # efficientnet 모델 배치사이드를 좀 많이 낮춰야함
    # net = models.efficientnet_b4(pretrained=True)
    # net.classifier[1] = nn.Linear(in_features=2560,out_features=450)#450개로 분류하잖음
    # net.to(device)

    # net = models.mobilenet_v2(pretrained=True)
    # net.head = nn.Linear(in_features=768, out_features=3)
    # net.to(device)

    # net = models.vgg19(pretrained=True)
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=450)
    # net.to(device)
    #
    # net = models.alexnet(pretrained=True)
    # net.head = nn.Linear(in_features=2060, out_features=450)
    # net.to(device)

    # net = models.densenet121(pretrained=True)
    # net.head = nn.Linear(in_features=2060, out_features=3)
    # net.to(device)

    # net = models.vision_transformer.VisionTransformer(pretrained=True)
    # net.head = nn.Linear(in_features=2060, out_features=3)
    # net.to(device)

    #### 4 epoch, optim loss
    # loss_function = LabelSmoothingCrossEntropy()
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.0001)
    # epochs = 20
    #
    # best_val_acc = 0.0
    #
    # train_steps = len(train_loader)
    # valid_steps = len(val_loader)
    # save_path = "best.pt"
    # dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
    #                              columns=["Epoch", "Accuracy"])
    # if os.path.exists(save_path):
    #     best_val_acc = max(pd.read_csv("./modelAccuracy.csv")["Accuracy"].tolist())
    #
    # for epoch in range(epochs):
    #     runing_loss = 0
    #     val_acc = 0
    #     train_acc = 0
    #
    #     net.train()
    #     train_bar = tqdm(train_loader, file=sys.stdout, colour='blue')
    #     for step, data in enumerate(train_bar):
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = net(images)
    #         loss = loss_function(outputs, labels)
    #
    #         optimizer.zero_grad()
    #         train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
    #         loss.backward()
    #         optimizer.step()
    #         runing_loss += loss.item()
    #         train_bar.desc = f"train epoch[{epoch + 1} / {epochs}], loss{loss.data:.3f}"
    #
    #     net.eval()
    #     with torch.no_grad():
    #         valid_bar = tqdm(val_loader, file=sys.stdout, colour='red')
    #         for data in valid_bar:
    #             images, labels = data
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = net(images)
    #             val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
    #
    #     val_accuracy = val_acc / len(val_dataset)
    #     train_accuracy = train_acc / len(train_dataset)
    #
    #     dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
    #     dfForAccuracy.loc[epoch, 'Accuracy'] = round(val_accuracy, 3)
    #     print(
    #         f"epoch[{epoch + 1}/{epochs}]"f"train loss :{(runing_loss / train_steps):.3f}"f"train_acc: {train_accuracy:.3f} val_acc: {val_accuracy:.3f}")
    #     if val_accuracy > best_val_acc:
    #         best_val_acc = val_accuracy
    #         torch.save(net.state_dict(), save_path)
    #
    #     if epoch == epochs - 1:
    #         dfForAccuracy.to_csv("./modelAccuracy.csv", index_label=False)
    #
    # torch.save(net.state_dict(), "./last.pt")


    #loss_function = LabelSmoothingCrossEntropy()
    loss_function = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    epochs = 50

    best_val_acc = 0.0

    train_steps = len(train_loader)
    valid_steps = len(val_loader)
    save_path = "lastmodel.pt"
    dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                 columns=['Epoch', 'Accuracy'])

    if os.path.exists(save_path) :
        best_val_acc = max(pd.read_csv('./modelAccuracy.csv')['Accuracy'].tolist())
        net.load_state_dict(torch.load(save_path))

    for epoch in range(epochs) :
        runing_loss = 0
        val_acc = 0
        train_acc = 0

        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='blue')
        for step, data in enumerate(train_bar) :
            images , labels = data
            images , labels = images.float().to(device) , labels.to(device)
            # print(images.shape)
            # print(net.conv1)
            # exit()
            outputs = net(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            runing_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch+1} / {epochs}], loss{loss.data:.3f}"

        net.eval()
        with torch.no_grad() :
            valid_bar = tqdm(val_loader, file=sys.stdout, colour='red')
            for data in valid_bar :
                images, labels = data
                images, labels = images.float().to(device), labels.to(device)
                outputs = net(images)
                val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        val_accuracy = val_acc / len(val_dataset)
        train_accuracy = train_acc / len(train_dataset)

        dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
        dfForAccuracy.loc[epoch, 'Accuracy'] = round(val_accuracy, 3)
        print(f"epoch [{epoch+1}/{epochs}]"
              f" train loss : {(runing_loss / train_steps):.3f} "
              f"train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}"
        )

        if val_accuracy > best_val_acc :
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        if epoch == epochs - 1 :
            dfForAccuracy.to_csv("./modelAccuracy.csv" , index=False)


if __name__ == '__main__':
    main()
