# Purpose : anlyse abwalk with new static algorithm
import cv2
from cvzone.PoseModule import PoseDetector
import numpy as np
import os


def poseAnalyser(video):
    videoWidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    videoHeight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    videoFps = video.get(cv2.CAP_PROP_FPS)
    print(notices[langType][1], end="")
    print(f"{videoWidth}px * {videoHeight}px, {videoFps}FPS")

    # importing PoseModel detector
    detector = PoseDetector()

    # Calculate hyperparameters
    dt = 1 / videoFps

    while True:
        try:
            ret, img = video.read()

            if ret:
                img = detector.findPose(img, False)
                lmList, _ = detector.findPosition(img, False)

                # A man has been detected
                if len(lmList) != 0:
                    circleTargets = (0, *range(11, 33))

                    for i in circleTargets:
                        dotColor = (0, 0, 255) if i == 0 else (255, 0, 0)
                        newCoordinates = (lmList[i][1], lmList[i][2])
                        img = cv2.ellipse(
                            img,
                            newCoordinates,
                            (5, 5),
                            0,
                            0,
                            360,
                            dotColor,
                            -1,
                        )

                        # For Delveopment, put text and draw the background
                        # rectangleStart = (newCoordinates[0] + 5, newCoordinates[1] - 7)
                        # rectangleEnd = (rectangleStart[0] + 20, rectangleStart[1] + 14)
                        # img = cv2.rectangle(
                        #     img, rectangleStart, rectangleEnd, (255, 255, 255), -1
                        # )
                        # textStart = (newCoordinates[0] + 5, newCoordinates[1] + 3)
                        # img = cv2.putText(
                        #     img,
                        #     f"{i}",
                        #     textStart,
                        #     cv2.FONT_HERSHEY_PLAIN,
                        #     1,
                        #     (0, 0, 0),
                        #     1,
                        # )

                cv2.imshow("img", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        except Exception as e:
            print(f"[Error]{e}")

    return True


def main():
    cam = cv2.VideoCapture(samplePath)

    isAnalysedDirExist = os.path.isdir(analysedDirPath)

    if not isAnalysedDirExist:
        print(notices[langType][0])
        os.mkdir(analysedDirPath)

    poseAnalyser(cam)

    return True


# mockup data
samplePath = "./source/01-00.mp4"

# i18n
notices = {
    "en": [
        "Looks like there is no '/analysed/' directory. We'll generate it...",
        "Video has been loaded.",
    ],
    "kr": [
        "'analysed' 디렉토리가 존재하지 않는 것 같습니다. 생성합니다...",
        "비디오가 성공적으로 로딩되었습니다.",
    ],
}
langType = "en"

# configurations
analysedDirPath = "./analysed"

if __name__ == "__main__":
    main()
