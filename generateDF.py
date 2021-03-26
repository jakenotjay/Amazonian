import csv
import pandas as pd

filename = "./data/AMZNweekly.csv"

dates = []
openList = []
high = []
low = []
close = []
adjusted_close = []
volume = []

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            dates.append(row[0])
            openList.append(row[1])
            high.append(row[2])
            low.append(row[3])
            close.append(row[4])
            adjusted_close.append(row[5])
            volume.append(row[6])
            line_count += 1

df = pd.DataFrame(list(zip(dates, openList, high, low, close, adjusted_close, volume)), columns=['Dates','Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

print(df)

df.to_pickle('AMZNweekly.pkl')