import cv2
import time
import numpy as np
import torch
import torch.nn
from torchvision import models, transforms
import os, sys
import traceback
from PIL import Image


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


if __name__ == '__main__':
    video_path = 'video/snare.mp4'
    model_path = 'model/classified_1012.pth'
    save_path = 'output'

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
    seconds = 0
    minutes = 0
    doc_time = {'anus':False, 'blank':0,'cecum':False, 'forceps':False, 'snare':False, 'clip':False}
    frame_count = {'anus':0, 'blank':0,'cecum':0, 'forceps':0, 'snare':0, 'clip':0}
    frame_label = {'anus':0, 'blank':0,'cecum':0, 'forceps':0, 'snare':0, 'clip':0}
    anus,cecum,forceps,snare,clip = 0,0,0,0,0
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            frame_copy = frame.copy()
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            frame_copy = Image.fromarray(frame_copy)
            inputs = preprocess(frame_copy).unsqueeze(0).to(device)
            label = detector(inputs, model)
            #   Time counter
            seconds = int(time.time() - time_start) - minutes * 60
            if seconds >= 60:
                    minutes += 1
                    seconds = 0
            m = "%02d" %minutes
            s = "%02d" %seconds
            timestamp = "{}:{}".format(m,s)
            cv2.putText(frame, timestamp, (100, 100), cv2.FONT_ITALIC, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
        
            if frame_label[label]==0:
                frame_label[label]=frameID
            elif frame_label[label] == frameID-1:
                frame_label[label]=frameID
                frame_count[label] = frame_count[label]+1
            elif frame_label[label] != frameID-1:
                frame_count[label] = 0
                frame_label[label] = 0

            if doc_time[label]==False and frame_count[label]>60:
                doc_time[label]=True

            
            if doc_time['cecum']==True:
                cecum+=1
                if cecum <= 600:
                    text = '      cecum detected'
                    cv2.putText(frame, text, (100, 100), cv2.FONT_ITALIC, 
                            1, (127, 255, 0), 2, cv2.LINE_AA)
                elif cecum > 600:
                    doc_time['cecum']='Close'
            elif doc_time['anus']==True:
                anus+=1
                if anus <= 600:
                    text = '      anal detected'
                    cv2.putText(frame, text, (100, 100), cv2.FONT_ITALIC, 
                        1, (127, 255, 0), 2, cv2.LINE_AA)
                elif anus > 600:
                    doc_time['anus']='Close'


            if doc_time['forceps']==True:
                forceps+=1
                if forceps <= 600:
                    text = 'forceps insert'
                    cv2.putText(frame, text, (100, 150), cv2.FONT_ITALIC, 
                            1, (127, 255, 0), 2, cv2.LINE_AA)
                elif forceps > 1200:
                    doc_time['forceps']=False
            elif doc_time['snare']==True:
                snare+=1
                if snare <= 600:
                    text = 'snare insert'
                    cv2.putText(frame, text, (100, 150), cv2.FONT_ITALIC, 
                            1, (127, 255, 0), 2, cv2.LINE_AA)
                elif snare > 1200:
                    doc_time['snare']=False
            elif doc_time['clip']==True:
                clip+=1
                if clip <= 600:
                    text = 'clip insert'
                    cv2.putText(frame, text, (100, 150), cv2.FONT_ITALIC, 
                            1, (127, 255, 0), 2, cv2.LINE_AA)
                elif clip > 1200:
                    doc_time['clip']=False
            
            frameID += 1
            # 寫入 影片
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
