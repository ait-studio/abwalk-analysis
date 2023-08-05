import csv
import numpy as np


def main():
    right_arm_angle = []
    left_arm_angle = []
    left_waist_angle = []
    right_waist_angle = []
    step_length = []
    step_speed = []
    bent_angle = []
    heights = []

    with open(f"{csvFilePath}/{smapleFileName}", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            right_arm_angle.append(row["right_arm_angle"])
            left_arm_angle.append(row["left_arm_angle"])
            left_waist_angle.append(row["left_waist_angle"])
            right_waist_angle.append(row["right_waist_angle"])
            step_length.append(row["step_length"])
            step_speed.append(row["step_speed"])
            bent_angle.append(row["bent_angle"])
            heights.append(row["height"])

    sliceSize = 50
    sliceCount = int(len(right_arm_angle) / sliceSize)
    sliceCount = (
        sliceCount
        if sliceCount * 50 == len(right_arm_angle)
        else sliceCount + 1
        # If the length is the exactly multiplied number of {sliceCount}, we don't need to plus 1
    )

    # string to float
    dataArray = [
        right_arm_angle,
        left_arm_angle,
        left_waist_angle,
        right_waist_angle,
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

            curAngleDiff = round((curAngles[-3] - curAngles[2]), 2)

            extractedArmAngles.append(curAngleDiff)

        averageArmAngle = np.average(extractedArmAngles)
        if idx == 0:
            print("Right Arm Angle AVG", end="\t")
        else:
            print("Left Arm Angle AVG", end="\t")
        print(f"{round(averageArmAngle, 2)}°")

    # Calculate Height-Pixel constant
    meanHightPixels = np.average(heights[:10])  # pick only initial 10 values
    realHeight = 170
    heightFactor = 0.94  # nose node to toe doesn't mean the 100% of the height

    heightPixelConstant = round((realHeight * heightFactor / meanHightPixels), 2)
    print(f"Height Pixcel Constant\t{heightPixelConstant}cm/px")

    # step length
    new_step_length = sorted(step_length)
    extractedStepLength = new_step_length[-10] * heightPixelConstant
    print(f"Extracted Step length \t{round(extractedStepLength, 2)}cm")

    sortedAvgStepSpeed = sorted(step_speed)
    selectedStepSpeed = []

    if len(sortedAvgStepSpeed) >= 10:
        selectedStepSpeed.append(sortedAvgStepSpeed[-10])
    else:
        print("Can't get three highest value of step lengths data")

    avgSelectedStepSpeed = np.average(selectedStepSpeed)
    unitConvertingConstant = 0.036  # c(m)ps to kph
    stepSpeedFactor = 0.95
    avgSelectedStepSpeed = (
        avgSelectedStepSpeed
        * stepSpeedFactor
        * unitConvertingConstant
        * heightPixelConstant
    )

    print(f"Extracted Step Speed \t{round(avgSelectedStepSpeed, 2)}kph")

    # Step Asymmetry
    avgLeftWaist = np.average(left_waist_angle)
    avgRightWaist = np.average(right_waist_angle)
    diffWaist = abs(avgLeftWaist - avgRightWaist)
    sumWaist = abs(avgLeftWaist + avgRightWaist)
    asymmetryAmount = round(abs(diffWaist / sumWaist) * 100, 4)

    print(f"Step Asymmetry is \t{asymmetryAmount}%")

    # Torso Bented
    avgBented = round(np.average(bent_angle), 4)
    print(f"Torso Bented amount is \t{avgBented}Degree")

    return True


csvFilePath = "./analysed"
smapleFileName = "01-90.mp4.csv"


if __name__ == "__main__":
    main()
