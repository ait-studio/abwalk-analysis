import cv2
import numpy as np
import os

def main():
    net = cv2.dnn.readNet("./model/yolov3.weights", "./model/yolov3.cfg")
    classes = []

    with open("./model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_name = net.getLayerNames()
    # print(layer_name)
    output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
    # print(output_layer)
    colors = np.random.uniform(0, 255, size = (len(classes), 3))

    listDir = os.listdir("./source")
    listDir = listDir[:1]

    for i in listDir:
        fileName, fileType = i.split(".")
        # print(fileName, " ", fileType)
        
        if fileType != "mp4" :
            continue

        print(f"Start to convert {i}")
        cap = cv2.VideoCapture(f"./source/{i}")

        frameRate = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        outFileName = "./extracted/" + fileName + ".avi"
        outFile = cv2.VideoWriter(outFileName, fourcc, frameRate, (1920, 1080))

        curFrame = 0
        barWidth = 50

        while True:
            ret, img = cap.read()
            width, height, channel = img.shape

            if ret:
                img = cv2.resize(img, None, fx=0.4, fy=0.4)

                blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layer)

                # Showing Infos on the screen
                class_ids = []
                confidences = []
                boxes = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            centerX = int(detection[0] * width)
                            centerY = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            x = int(centerX - w/2)
                            y = int(centerY - h/2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                # print(indexes)

                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])

                        color = colors[i]
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        img = cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

                outFile.write(img)
                # cv2.imshow("img", img)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                curFrame += 1
                percentage = int((curFrame / frameCount )* barWidth)
                print("Progress [", "█" * percentage, " " * (barWidth - percentage), f"] {curFrame} / {frameCount}", end="\r")
            else :
                break
        
        print("")
        cap.release()
        out.release()

    return

if __name__ == "__main__":
    main()

print("Program quit")
