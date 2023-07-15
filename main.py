import cv2
import numpy as np
import os

def main():
    print("Select converting mode please : ")
    weightDirectory = "./model/weights"
    cfgDirectory = "./model/cfgs"

    while True:
        print("Program quit with enter -1")
        choice = int(input("1. Standard mode\t2. Light mode(use tiny weights and cfg)\n"))

        if choice == 1:
            net = cv2.dnn.readNet(f"{weightDirectory}/yolov3.weights", f"{cfgDirectory}/yolov3.cfg")
            break
        elif choice == 2:
            net = cv2.dnn.readNet(f"{weightDirectory}/yolov3-tiny.weights", f"{cfgDirectory}/yolov3-tiny.cfg")
            break
        elif choice == -1:
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
    listDir = ["01-00.mp4"]

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

        while True:
            ret, img = cap.read()

            if ret:
                # img = cv2.resize(img, None, fx=0.4, fy=0.4)
                height, width, _ = img.shape

                # blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
                        # print(x, y, w, h)
                        label = str(classes[class_ids[i]])

                        color = (255, 125, 0)
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        img = cv2.putText(img, label, (x, y - 30), font, 2, color, 2)

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
        outFile.release()

    return

if __name__ == "__main__":
    main()

print("Program quit")
