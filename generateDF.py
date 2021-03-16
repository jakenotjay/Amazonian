import csv
import pandas as pd

filename = "./data/AMZN.csv"

dates = []
adjusted_close = []

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            dates.append(row[0])
            adjusted_close.append(row[5])
            line_count += 1

df = pd.DataFrame(list(zip(dates, adjusted_close)), columns=['Dates', 'Adj Close'])

print(df)

df.to_pickle('AMZN.pkl')