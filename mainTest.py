import cv2
import numpy as np
import os

def main():
    net = cv2.dnn.readNet("./model/yolov3.weights", "./model/yolov3.cfg")
    classes = []

    with open("./model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_name = net.getLayerNames()
    output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size = (len(classes), 3))

    listDir = os.listdir("./source")
    listDir = ["pkWalkTest2.mp4"]

    for i in listDir:
        fileName, fileType = i.split(".")
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
        
        # For ROI video
        standardSize = (600, 800) # Define the standard size for the bounding boxes
        roiOutFileName = "./extracted/" + fileName + "_roi.avi"
        roiOutFile = cv2.VideoWriter(roiOutFileName, fourcc, frameRate, standardSize)

        curFrame = 0
        barWidth = 50

        while True:
            ret, img = cap.read()

            if ret:
                height, width, _ = img.shape

                blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layer)

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
                            w = int(detection[2] * width * 1.2) # Increased by 20%
                            h = int(detection[3] * height * 1.2) # Increased by 20%

                            x = int(centerX - w/2)
                            y = int(centerY - h/2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])

                        color = colors[i]
                        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # img = cv2.putText(img, label, (x, y - 30), font, 2, color, 2)
                        
                        # Extract ROI and write to new video file
                        roi = img[y:y+h, x:x+w]
                        roi_resized = cv2.resize(roi, standardSize) # Resize to match the standard size while maintaining aspect ratio
                        roiOutFile.write(roi_resized)

                outFile.write(img)

                curFrame += 1
                percentage = int((curFrame / frameCount )* barWidth)
                print("Progress [", "█" * percentage, " " * (barWidth - percentage), f"] {curFrame} / {frameCount}", end="\r")
            else :
                break
        
        print("")
        cap.release()
        outFile.release()
        roiOutFile.release() # Release the ROI video writer

    return

if __name__ == "__main__":
    main()

print("Program quit")

