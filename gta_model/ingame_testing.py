from torchvision import models
import torch
import torch.nn as nn
from PIL import ImageGrab
import cv2
import torch.nn.functional as F
import albumentations as Al
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from input_keys import PressKey, ReleaseKey
from Lane_use import process_img

labels = {0: 'a', 1: 'w', 2: 'd'}#, 3: 's'}


# test transform 지정
test_transform = Al.Compose(
    [
        Al.Resize(width = 480, height = 270),
        Al.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 호출
net = models.mobilenet_v3_large(pretrained=True)    # mobilenet-v3
net.classifier[3] = nn.Linear(in_features = 1280, out_features=3)
net.load_state_dict(torch.load('./models/mbv3model_3features.pt', map_location=device))


# net = models.efficientnet_b4(pretrained=True)   # efficientnet
# net.classifier[1] = nn.Linear(in_features=1792,out_features=4)
# net.load_state_dict(torch.load('./models/test2model.pt', map_location=device))
#
#
# net = models.resnet18(pretrained=True)
# #net.conv1 = nn.Linear(net.conv1.in_features, 64*3*7*7)
# net.fc = nn.Linear(in_features=512,out_features=4)#450개로 분류하잖음
# net.load_state_dict(torch.load('./models/resnet18.pt', map_location=device))

# net = models.densenet121(pretrained=True)
# net.classifier = nn.Linear(in_features=1024, out_features=3)
# net.load_state_dict(torch.load('./models/densenet_3feature.pt', map_location=device))

net.to(device)
net.eval()

def ingame_predic():
    while(True):
        with torch.no_grad():

            screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 760)))    # 화면의 일부(1280, 720)을 받아서 Numpy array로 저장
            # screen = np.array(Image.open('./test_images/6.jpg').resize((1280, 720)))    # test image 이용시
            input_image = test_transform(image=screen)['image'].float().unsqueeze(0).to(device) # test transform 적용

            output = net(input_image)   # 모델에서 predict 결과 받음
            softmax_result = F.softmax(output)  # 결과값을 softmax를 거쳐 확률값으로 변경

            new_screen, original_image, m1, m2, minimap,minim = process_img(screen) # 이미지에서 차선 검출과 미니맵 검출에 필요한 전처리 진행

            cv2.imshow('mini',cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB)) # 미니맵 이미지 확인

            # 차선 검출로 인식한 차선의 기울기를 통해 기존 predict 확률 보정
            if m1 < 0 and m2 < 0:
                print('line talk to me, go right')
                softmax_result = softmax_result + torch.tensor([[-0.0, -0.0, 0.25]]).to(device) #right
            elif m1 > 0 and m2 > 0:
                print('line talk to me, go left')
                softmax_result = softmax_result + torch.tensor([[0.25, -0.0, -0.0]]).to(device) #left
            else:
                print('line talk to me, go foward')
                softmax_result = softmax_result + torch.tensor([[-0.0, 0.35, -0.0]]).to(device) #straight
                
            print(minim)    # 미니맵에서 검출한 기울기 출력
            # if minim<0:
            #     print('map talk to me, go right')
            #     softmax_result = softmax_result + torch.tensor([[-0.1, -0.1, 0.2]]).to(device) #right
            # elif minim > 0:
            #     print('map talk to me, go left')
            #     softmax_result = softmax_result + torch.tensor([[0.2, -0.1, -0.1]]).to(device) #left
            # else:
            #     print('map talk to me, go foward')
            #     softmax_result = softmax_result + torch.tensor([[-0.1, 0.2, -0.1]]).to(device)

            # predict한 최고 확률과 그 라벨값을 받아줌
            top_prob, top_label = torch.topk(softmax_result, 1)
            prob = round(top_prob.item() * 100, 2)
            label = labels.get(int(top_label))

            # print(f'prob: {prob}, label: {label}')


            # 게임 내 키입력을 위해 windows에서 각 키의 값을 저장
            W = 0x11
            A = 0x1E
            S = 0x1F
            D = 0x20
            T = 0x14

            # predict와 보정을 거친 값을 기준으로 지정된 정책에 따라 진행 방향 결정
            if (65 < prob) and (label == 'a'):

                PressKey(W)
                PressKey(A)
                ReleaseKey(S)
                ReleaseKey(D)
                # time.sleep(0.7)
                # ReleaseKey(W)
                # ReleaseKey(A)


            elif (65 < prob) and (label == 'w'):
                PressKey(W)
                ReleaseKey(A)
                ReleaseKey(S)
                ReleaseKey(D)
                # time.sleep(0.7)
                # ReleaseKey(W)
            # elif (65 < prob < 80) and (label == 'a'):
            #
            #     PressKey(A)
            #     ReleaseKey(S)
            #     ReleaseKey(D)
            #     ReleaseKey(W)
            #     # #ReleaseKey(W)
            #     # time.sleep(0.3)
            #     # ReleaseKey(A)


            # elif (65 < prob < 80) and (label == 'd'):
            #     PressKey(D)
            #     ReleaseKey(A)
            #     ReleaseKey(S)
            #     ReleaseKey(W)
            #     # time.sleep(0.7)
            #     # ReleaseKey(D)
            elif (65 < prob) and (label == 'd'):
                PressKey(W)
                PressKey(D)
                ReleaseKey(A)
                ReleaseKey(S)
                #ReleaseKey(W)



            # elif (50 < prob) and (label == 's'):
            #     PressKey(S)


            else:
                PressKey(S)
                ReleaseKey(A)
                ReleaseKey(D)
                ReleaseKey(W)


        print(prob, label)
    #return prob, label


if __name__ == '__main__':
    ingame_predic()