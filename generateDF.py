import csv
import pandas as pd

filename = "./data/QCOMdaily.csv"

dates = []
openList = []
high = []
low = []
close = []
adjusted_close = []
volume = []

# dates = []
# trends = []

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {",".join(row)}')
            line_count += 1
        else:
            splitDates = row[0].split('/')
            # dates.append(date(year=int(splitDates[2]), month=int(splitDates[1]), day=int(splitDates[0])))
            if(row[0] != 'null' and row[1] != 'null' and row[2] != 'null' and row[3] != 'null' and row[4] != 'null' and row[5] != 'null' and row[6] != 'null'):
                dates.append(row[0])
            # trends.append(row[1])
                openList.append(row[1])
                high.append(row[2])
                low.append(row[3])
                close.append(row[4])
                adjusted_close.append(row[5])
                volume.append(row[6])
            line_count += 1

df = pd.DataFrame(list(zip(dates, openList, high, low, close, adjusted_close, volume)), columns=['Dates','Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
print(df)
df['Dates'] = pd.to_datetime(df['Dates'])
df = df.set_index('Dates')

print(df)
print(df.dtypes)

df.to_pickle('QCOMdaily.pkl')