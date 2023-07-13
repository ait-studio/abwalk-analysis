import cv2
import os

def main():
    listDir = os.listdir("./src")

    for i in listDir:
        fileName, fileType = i.split(".")
        # print(fileName, " ", fileType)
        
        if fileType != "mp4" :
            continue

        print(f"Start to convert {i}")
        cap = cv2.VideoCapture(f"./src/{i}")

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frameRate = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        outFileName = "./converted/" + fileName + ".avi"
        out = cv2.VideoWriter(outFileName, fourcc, frameRate, (1920, 1080))

        curFrame = 0
        barWidth = 50

        while True:
            ret, img = cap.read()

            if ret:
                img = cv2.resize(img, (1920, 1080))
                out.write(img)
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
