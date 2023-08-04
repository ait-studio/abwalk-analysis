import csv
import numpy as np


def main():
    right_arm_angle = []
    left_arm_angle = []
    left_wrist_angle = []
    right_wrist_angle = []
    step_length = []
    step_speed = []
    bent_angle = []

    with open(f"{csvFilePath}/{smapleFileName}", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            right_arm_angle.append(row["right_arm_angle"])
            left_arm_angle.append(row["left_arm_angle"])
            left_wrist_angle.append(row["left_wrist_angle"])
            right_wrist_angle.append(row["right_wrist_angle"])
            step_length.append(row["step_length"])
            step_speed.append(row["step_speed"])
            bent_angle.append(row["bent_angle"])

    sliceSize = 50
    sliceCount = int(len(right_arm_angle) / sliceSize) + 1

    targetDataset = [right_arm_angle, left_arm_angle]

    for idx, data in enumerate(targetDataset):
        extractedArmAngles = []
        for i in range(sliceCount):
            startIndex = i * 50
            endIndex = (i + 1) * 50 if (i + 1) * 50 < len(data) else len(data)
            # print(startIndex, endIndex)

            curAngles = data[startIndex:endIndex]
            curAngles = sorted(curAngles)
            curAngleDiff = round((float(curAngles[2]) - float(curAngles[-2])), 2)

            extractedArmAngles.append(curAngleDiff)

        averageArmAngle = np.average(extractedArmAngles)
        if idx == 0:
            print("Right Arm Angle AVG", end="\t")
        else:
            print("Left Arm Angle AVG", end="\t")
        print(averageArmAngle)

    return True


csvFilePath = "./analysed"
smapleFileName = "01-90.mp4.csv"


if __name__ == "__main__":
    main()
