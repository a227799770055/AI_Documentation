import cv2
import time
from cv2 import threshold
import numpy as np
import torch
import torch.nn
from torchvision import models, transforms
import os, sys
import traceback
from PIL import Image
from utils.opt import video_parse_opt

def detector(inputs, model):
    #   Class labels for prediction
    class_names = ['anus', 'blank','cecum', 'forceps', 'snare', 'clip']
    # class_names = ['anus','cecum', 'forceps', 'snare', 'clip']
    soft = torch.nn.Softmax(dim=1)
    outputs = model(inputs)
    out = soft(outputs)
    out = out.cpu().detach().numpy()
    out = out[0]>0.95
    pred = np.where(out==True)[0]
    
    if len(pred) != 0:
        label = class_names[pred[0]]
    else:
        label = 'blank'
    print(label)
    return label 

def time_counter(start, minutes, seconds):
    seconds = int(time.time() - start) - minutes * 60
    if seconds >= 60:
        minutes += 1
        seconds = 0
    mins = "%02d" %minutes
    secs = "%02d" %seconds
    return mins, secs


if __name__ == '__main__':

    opt = video_parse_opt()

    video_path = opt.video_path
    model_path = opt.model_path
    save_path = opt.save_path

    #   load model 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path,map_location=device)
    model = model.eval()


    #   frame process
    preprocess=transforms.Compose([
        transforms.Resize(size=512),
        transforms.CenterCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    #   loading video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_name = (video_path.split('/')[-1]).split('.')[0]
    out = cv2.VideoWriter(os.path.join(save_path, '{}.mp4'.format(video_name)), fourcc, fps, (width, height))

    frameID = 0
    time_start = time.time()
    seconds, minutes = 0, 0
    threshold = {'anus':False, 'blank':0,'cecum':False, 'forceps':False, 'snare':False, 'clip':False}
    frame_count = {'anus':0, 'blank':0,'cecum':0, 'forceps':0, 'snare':0, 'clip':0}
    frame_label = {'anus':0, 'blank':0,'cecum':0, 'forceps':0, 'snare':0, 'clip':0}
    timer_label = {'anus':0, 'blank':0,'cecum':0, 'forceps':0, 'snare':0, 'clip':0}
    # anus,cecum,forceps,snare,clip = 0,0,0,0,0
    
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            frame_copy = frame.copy()
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            frame_copy = Image.fromarray(frame_copy)
            inputs = preprocess(frame_copy).unsqueeze(0).to(device)
            label = detector(inputs, model)
            
            # Time counter 計算影片時長
            mins, secs = time_counter(time_start, minutes, seconds)
            timestamp = "{}:{}".format(mins, secs)
            cv2.putText(frame, timestamp, (100, 100), cv2.FONT_ITALIC, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
        
            # 計算label連續出現的幀數，當沒有連貫時，幀數歸0
            # frame_label 紀錄上一幀的位置
            # frame_count 紀錄連續出現幾幀
            if frame_label[label]==0:
                frame_label[label]=frameID
            elif frame_label[label] == frameID-1:
                frame_label[label]=frameID
                frame_count[label] = frame_count[label]+1
            elif frame_label[label] != frameID-1:
                frame_count[label] = 0
                frame_label[label] = 0

            # 當特定label的幀數連續出現且達到閥值時，該label的閥門就會變成true
            if threshold[label]==False and frame_count[label]>60: # threshold setting as 60 frames
                threshold[label]=True

            # 當閥門打開後，就會顯示偵測到該label
            # 當顯示一定的時間後，閥門會在關閉
            labels_key = threshold.keys()
            for key in labels_key:
                if threshold[key] == True:
                    timer_label[key] = timer_label[key] + 1
                    if key == 'cecum' or key == 'anus':
                        if timer_label[key] <= 600:
                            text = '      {} detected'.format(key)
                            cv2.putText(frame, text, (100, 100), cv2.FONT_ITALIC, 
                                        1, (127, 255, 0), 2, cv2.LINE_AA)
                        elif timer_label[key] > 600:
                            threshold[key]='Close'
                    elif key == 'forceps' or key == 'snare' or key == 'clip':
                        if timer_label[key] <= 600:
                            text = '{} insert'.format(key)
                            cv2.putText(frame, text, (100, 150), cv2.FONT_ITALIC, 
                                        1, (127, 255, 0), 2, cv2.LINE_AA)
                        elif timer_label[key] > 1200: #為了避免關閉閥門後 短時間的偵測造成閥門再次開啟 將閥門關閉時間拉長
                            threshold[key] == False

            frameID += 1
            # 寫入影片
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            error_class = e.__class__.__name__ #取得錯誤類型
            detail = e.args[0] #取得詳細內容
            cl, exc, tb = sys.exc_info() #取得Call Stack
            lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
            fileName = lastCallStack[0] #取得發生的檔案名稱
            lineNum = lastCallStack[1] #取得發生的行號
            funcName = lastCallStack[2] #取得發生的函數名稱
            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            print(errMsg)
            break

    cap.release()
    cv2.destroyAllWindows()
