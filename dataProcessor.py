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
    heights = []

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
            heights.append(row["height"])

    sliceSize = 50
    sliceCount = int(len(right_arm_angle) / sliceSize) + 1

    # string to float
    dataArray = [
        right_arm_angle,
        left_arm_angle,
        left_wrist_angle,
        right_wrist_angle,
        step_length,
        step_speed,
        bent_angle,
        heights,
    ]

    for data in dataArray:
        for idx, row in enumerate(data):
            data[idx] = float(row)

    # arm angle average

    targetDataset = [right_arm_angle, left_arm_angle]

    for idx, data in enumerate(targetDataset):
        extractedArmAngles = []
        for i in range(sliceCount):
            startIndex = i * 50
            endIndex = (i + 1) * 50 if (i + 1) * 50 < len(data) else len(data)
            # print(startIndex, endIndex)

            curAngles = data[startIndex:endIndex]
            curAngles = sorted(curAngles)

            curAngleDiff = round((curAngles[2] - curAngles[-2]), 2)

            extractedArmAngles.append(curAngleDiff)

        averageArmAngle = np.average(extractedArmAngles)
        if idx == 0:
            print("Right Arm Angle AVG", end="\t")
        else:
            print("Left Arm Angle AVG", end="\t")
        print(f"{round(averageArmAngle, 4)}Degree")

    # step length
    new_step_length = sorted(step_length)
    extractedStepLength = new_step_length[-10] * 0.35
    print(f"Extracted Step length \t {round(extractedStepLength, 4)}cm")

    # step speed
    sliceSize = 10
    sliceCount = int(len(right_arm_angle) / sliceSize) + 1

    avgStepSpeeds = []

    for i in range(sliceCount):
        startIndex = i * 10
        endIndex = (i + 1) * 10 if (i + 1) * 10 < len(data) else len(data)

        curStepSpeed = step_speed[startIndex:endIndex]
        avgStepSpeeds.append(np.average(curStepSpeed))

    sortedAvgStepSpeed = sorted(avgStepSpeeds)
    selectedStepSpeed = []

    if len(sortedAvgStepSpeed) >= 10:
        selectedStepSpeed.append(sortedAvgStepSpeed[-8])
        selectedStepSpeed.append(sortedAvgStepSpeed[-9])
        selectedStepSpeed.append(sortedAvgStepSpeed[-10])
    elif len(sortedAvgStepSpeed) >= 3:
        selectedStepSpeed.append(sortedAvgStepSpeed[0])
        selectedStepSpeed.append(sortedAvgStepSpeed[1])
        selectedStepSpeed.append(sortedAvgStepSpeed[2])
    else:
        print("Can't get three highest value of step lengths data")

    avgSelectedStepSpeed = np.average(selectedStepSpeed)
    unitConvertingConstant = 0.036  # cm/s to kph
    stepSpeedFactor = 0.95
    avgSelectedStepSpeed = (
        avgSelectedStepSpeed * stepSpeedFactor * unitConvertingConstant
    )

    print(f"Extracted Step Speed \t {round(avgSelectedStepSpeed, 4)}kph")

    # Step Asymmetry
    avgLeftWrist = np.average(left_wrist_angle)
    avgRightWrist = np.average(right_wrist_angle)
    diffWrist = abs(avgLeftWrist - avgRightWrist)
    sumWrist = abs(avgLeftWrist + avgRightWrist)
    asymmetryAmount = round(abs(diffWrist / sumWrist) * 100, 4)

    print(f"Step Asymmetry is \t {asymmetryAmount}%")

    # Torso Bented
    avgBented = round(np.average(bent_angle), 4)
    print(f"Torso Bented amount is \t {avgBented}Degree")

    return True


csvFilePath = "./analysed"
smapleFileName = "01-90.mp4.csv"


if __name__ == "__main__":
    main()
