# Purpose : anlyse abwalk with new static algorithm
import cv2
from cvzone.PoseModule import PoseDetector
import numpy as np
import os
import math
import csv


def getDistance(vec):
    xSquared = vec[0] ** 2
    ySquared = vec[1] ** 2
    dist = math.sqrt(xSquared + ySquared)

    return dist


def getAngle(vecA, vecB):
    dp = np.dot(vecA, vecB)
    norm1 = np.linalg.norm(vecA)
    norm2 = np.linalg.norm(vecB)
    radToDegree = 180 / math.pi
    curAngle = math.acos(dp / (norm1 * norm2)) * radToDegree
    return curAngle


def drawLine(coordinates, img, style=0):
    if style == 0:
        # Drawing Basic Skeleton Lines
        img = cv2.line(
            img,
            (coordinates[0], coordinates[1]),
            (coordinates[2], coordinates[3]),
            (0, 255, 0),
            2,
        )

    elif style == 1:
        # Drawing waist-shoulder lines
        img = cv2.line(
            img,
            (coordinates[0], coordinates[1]),
            (coordinates[2], coordinates[3]),
            (255, 0, 0),
            1,
        )

    return img


def poseAnalyser(video):
    videoWidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    videoHeight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    videoFps = video.get(cv2.CAP_PROP_FPS)
    videoFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print(notices[langType][1])
    print(f"{videoWidth}px * {videoHeight}px, {videoFps} FPS, {videoFrames} frames")

    # importing PoseModel detector
    detector = PoseDetector()

    # Calculate hyperparameters
    dt = 1 / videoFps

    lastWaistCoordi = [0, 0]

    curIdx = 0

    curData = np.zeros((1, 8))
    data = np.zeros((int(videoFrames), 8))
    barWidth = 50

    # outFileName = "./converted/" + fileName + ".mp4"
    outFileName = "./analysed/" + filename
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(
        outFileName, fourcc, videoFps, (int(videoWidth), int(videoHeight))
    )

    while True:
        try:
            ret, img = video.read()

            if ret:
                img = detector.findPose(img, False)
                lmList, _ = detector.findPosition(img, False)

                # A man has been detected
                if len(lmList) != 0:
                    circleTargets = (0, *range(11, 17), *range(23, 33))

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

                    lineTargets = [
                        (11, 12),  # Shoulder line
                        (12, 14),  # Right shoulder - Right elbow
                        (14, 16),  # Right elbow - Right wrist
                        (11, 13),  # Left shoulder - Left elbow
                        (13, 15),  # Left elbow - Left wrist
                        (12, 24),  # Right shoulder - Right waist
                        (11, 23),  # Left shoulder - Left waist
                        (24, 23),  # Left waist - Right waist
                        (24, 26),  # Right waist - Right knee
                        (23, 25),  # Left waist - Left knee
                        (26, 28),  # Right knee - Right ankle
                        (25, 27),  # Left knee - Left ankle
                        (28, 30),  # Right ankle - Right heel
                        (30, 32),  # Right heel - Right toe
                        (32, 28),  # Right toe - Right ankle
                        (27, 29),  # Left ankle - Left heel
                        (29, 31),  # Left heel - Left toe
                        (31, 27),  # Left toe - Left ankle
                    ]

                    for i in lineTargets:
                        img = drawLine(
                            (
                                lmList[i[0]][1],
                                lmList[i[0]][2],
                                lmList[i[1]][1],
                                lmList[i[1]][2],
                            ),
                            img,
                            0,
                        )

                    ## Get angles from here
                    angleTargets = [
                        (24, 12, 16),  # Right waist - shoulder - waist
                        (23, 11, 15),  # Left waist - shoulder - waist
                        (11, 23, 29),  # Left shoulder - waist - hill
                        (12, 24, 30),  # Right shoulder - waist - hill
                    ]

                    for idx, v in enumerate(angleTargets):
                        vecA = (
                            lmList[v[0]][1] - lmList[v[1]][1],
                            lmList[v[0]][2] - lmList[v[1]][2],
                        )
                        vecB = (
                            lmList[v[2]][1] - lmList[v[1]][1],
                            lmList[v[2]][2] - lmList[v[1]][2],
                        )
                        curAngle = getAngle(vecA, vecB)
                        curAngle = round(curAngle, 2)

                        curData[0][idx] = curAngle
                        # print(curAngle, end="\t")

                    distanceTagets = [(29, 30)]

                    for i in distanceTagets:
                        myVec = (
                            lmList[i[0]][1] - lmList[i[1]][1],
                            lmList[i[0]][2] - lmList[i[1]][2],
                        )

                        curDist = getDistance(myVec)
                        curDist = round(curDist, 2)

                        curData[0][4] = curDist
                        # print(curDist, end="\t")

                    if curIdx > 0 and curIdx % 10 == 0:
                        curWaistCoordi = [
                            lmList[23][1],  # Left waist x coordinate
                            lmList[24][1],  # Right waist x coordinate
                        ]
                        avgWaistCoordi = np.average(curWaistCoordi)
                        waistDiff = avgWaistCoordi - avgLastWaistCoordi
                        curKphPixel = waistDiff / (dt * 10)

                        curKphPixel = round(curKphPixel, 2)

                        curData[0][5] = curKphPixel
                        avgLastWaistCoordi = avgWaistCoordi
                        # print(curKphPixel, end="\t")
                    elif curIdx == 0:
                        curWaistCoordi = [
                            lmList[23][1],  # Left waist x coordinate
                            lmList[24][1],  # Right waist x coordinate
                        ]
                        avgWaistCoordi = np.average(curWaistCoordi)
                        avgLastWaistCoordi = avgWaistCoordi
                    else:
                        curData[0][5] = 0

                    avgPointShoulder = (
                        int((lmList[11][1] + lmList[12][1]) / 2),
                        int((lmList[11][2] + lmList[12][2]) / 2),
                    )
                    avgPointWaist = (
                        int((lmList[23][1] + lmList[24][1]) / 2),
                        int((lmList[23][2] + lmList[24][2]) / 2),
                    )
                    # img = cv2.line(img, avgPointShoulder, avgPointWaist, (0, 0, 255), 2)

                    torsoVector = (
                        avgPointShoulder[0] - avgPointWaist[0],
                        avgPointShoulder[1] - avgPointWaist[1],
                    )
                    verticalVector = (0, 1)

                    torsoBentAngle = getAngle(torsoVector, verticalVector)
                    torsoBentAngle = round(torsoBentAngle, 2)

                    curData[0][6] = torsoBentAngle
                    # print(torsoBentAngle, end="\t")

                    avgPointToe = (
                        int((lmList[29][1] + lmList[30][1]) / 2),
                        int((lmList[29][2] + lmList[30][2]) / 2),
                    )

                    heightVector = (
                        lmList[0][1] - avgPointToe[0],
                        lmList[0][2] - avgPointToe[1],
                    )

                    curHeight = round(np.linalg.norm(heightVector), 2)
                    curData[0][7] = curHeight

                    # print("")

                data = np.roll(data, -1, axis=0)
                data[-1] = curData[0]

                percentage = int((curIdx / videoFrames) * barWidth)
                print(
                    "Analysing [",
                    "█" * percentage,
                    " " * (barWidth - percentage),
                    f"] {curIdx} / {videoFrames}",
                    end="\r",
                )

                curIdx += 1
                out.write(img)
                # cv2.imshow("img", img)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            else:
                out.release()
                break
        except Exception as e:
            print(f"[Error]{e}")

    return data


def csvWriter(data):
    with open(f"{analysedDirPath}/{filename}.csv", "w") as f:
        fieldNames = [
            "right_arm_angle",
            "left_arm_angle",
            "left_waist_angle",
            "right_waist_angle",
            "step_length",
            "step_speed",
            "bent_angle",
            "height",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldNames)

        writer.writeheader()

        curIdx = 0
        videoFrames = len(data)
        barWidth = 50

        for i in data:
            curIdx += 1
            percentage = int((curIdx / videoFrames) * barWidth)
            print(
                "Writing [",
                "█" * percentage,
                " " * (barWidth - percentage),
                f"] {curIdx} / {videoFrames}",
                end="\r",
            )
            newRow = {
                "right_arm_angle": i[0],
                "left_arm_angle": i[1],
                "left_waist_angle": i[2],
                "right_waist_angle": i[3],
                "step_length": i[4],
                "step_speed": i[5],
                "bent_angle": i[6],
                "height": i[7],
            }
            writer.writerow(newRow)
    return True


def main():
    cam = cv2.VideoCapture(samplePath)

    isAnalysedDirExist = os.path.isdir(analysedDirPath)

    if not isAnalysedDirExist:
        print(notices[langType][0])
        os.mkdir(analysedDirPath)

    extractedData = poseAnalyser(cam)
    csvWriter(extractedData)

    return True


# mockup data
filename = "01-90.mp4"
samplePath = f"./source/{filename}"

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
