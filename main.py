import cv2
import numpy as np
import os

def main():
    print("Select converting mode please : ")
    weightDirectory = "./model/weights"
    cfgDirectory = "./model/cfgs"

    while True:
        print("Program quit with enter -1")
        modeChoice = int(input("1. Standard mode\t2. Light mode(use tiny weights and cfg)\n"))

        if modeChoice == 1:
            net = cv2.dnn.readNet(f"{weightDirectory}/yolov3.weights", f"{cfgDirectory}/yolov3.cfg")
            break
        elif modeChoice == 2:
            net = cv2.dnn.readNet(f"{weightDirectory}/yolov3-tiny.weights", f"{cfgDirectory}/yolov3-tiny.cfg")
            break
        elif modeChoice == -1:
            print("\n\nProgram has been quit...\n\nGood bye.\n")
            return
        else:
            print("Incorrect choice. Try again please")
    
    sizeChoice = -1
    modelSize = -1
    if modeChoice == 1:
        print("Select the model size please")
        while True:
            print("Program quit with enter -1")
            sizeChoice = int(input("1. smallest(fast)\t2.middle(normal)\t3.biggest(slow)\n"))

            if sizeChoice == 1:
                modelSize = (320, 320)
                break
            elif sizeChoice == 2:
                modelSize = (416, 416)
                break
            elif sizeChoice == 3:
                modelSize = (608, 608)
                break
            elif sizeChoice == -1:
                print("\n\nProgram has been quit...\n\nGood bye.\n")
                return
            else:
                print("Incorrect choice. Try again please")

    classes = []
    with open("./model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_name = net.getLayerNames()
    # print(layer_name)
    output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
    # print(output_layer)
    colors = np.random.uniform(0, 255, size = (len(classes), 3))

    listDir = os.listdir("./source")

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

        frameWidth = int(1920 * 1)
        frameHeight = int(1080 * 1)
        outFile = cv2.VideoWriter(outFileName, fourcc, frameRate, (frameWidth, frameHeight))

        curFrame = 0
        barWidth = 50

        roi = np.zeros((1920, 1080, 3), dtype="uint8")
        outRoiName = "./extracted/" + fileName + "_cropped.avi"
        outRoi = cv2.VideoWriter(outRoiName, fourcc, frameRate, (300, 600))

        while True:
            ret, img = cap.read()

            if ret:
                # img = cv2.resize(img, None, fx=0.4, fy=0.4)
                height, width, _ = img.shape

                modelSize = (608, 608) if modelSize == -1 else modelSize
                # print(modelSize)
                blob = cv2.dnn.blobFromImage(img, 0.00392, modelSize, (0, 0, 0), True, crop=False)
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
                            ratio = 1.2
                            centerX = int(detection[0] * width)
                            centerY = int(detection[1] * height)
                            w = int(detection[2] * width * ratio)
                            h = int(detection[3] * height * ratio)

                            x = int(centerX - w/2)
                            y = int(centerY - h/2)

                            x = x if x > 0 else 0
                            y = y if y > 0 else 0

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                # print(indexes)

                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        # print(x, y, w, h)
                        label = str(classes[class_ids[i]])

                        # save the cropped image
                        newCropped = np.zeros((600, 300, 3), dtype="uint8")
                        roi = img[ y : y + h, x : x + w]

                        # cv2.imshow("img", roi)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                        if len(roi) > 0:
                            print(x, y, w, h, len(roi))
                            if(h / w < 2):
                                factor = 300 / w
                                h_interval = int(h * factor)
                                roi = cv2.resize(roi, (300, h_interval))
                                
                                startAt = 300 - int(h_interval / 2)
                                endAt = startAt + h_interval
                                newCropped[startAt : endAt, : ] = roi
                            else :
                                factor = 600 / h
                                w_interval = int(w * factor)
                                roi = cv2.resize(roi, (w_interval, 600))

                                startAt = 150 - int(w_interval / 2)
                                endAt = startAt + w_interval
                                newCropped[ : , startAt : endAt] = roi
                        
                        outRoi.write(newCropped)

                        color = (255, 125, 0)
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        img = cv2.putText(img, label, (x, y - 30), font, 2, color, 2)

                outFile.write(img)
                # cv2.imshow("img", img)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                curFrame += 1
                filled = int((curFrame / frameCount ) * barWidth)
                realPercentage = int((curFrame / frameCount) * 100)
                percentageDigit = 1 if realPercentage < 10 else (2 if realPercentage < 100 else 3)
                
                filled = filled - (percentageDigit + 1)
                filled = filled if filled > 0 else 0
                progress = f"\033[;30;47m{' ' * (filled)}{realPercentage}%\033[;;m{' ' * (barWidth - (filled + percentageDigit + 1))}"
                # print("Progress |", progress, f"| {curFrame:04} / {int(frameCount):04}", end="\r")
            else :
                break
        
        print("")
        cap.release()
        outFile.release()

    return

if __name__ == "__main__":
    main()

print("Program quit")
