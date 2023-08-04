import csv


def main():
    with open(f"{csvFilePath}/{smapleFileName}", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row)
    return True


csvFilePath = "./analysed"
smapleFileName = "01-90.mp4.csv"


if __name__ == "__main__":
    main()
