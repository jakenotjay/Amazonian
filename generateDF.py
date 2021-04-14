import csv
import pandas as pd
from datetime import date

filename = "./data/USD1MTD156N.csv"

dates = []
rates = []

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {",".join(row)}')
            line_count += 1
        else:
            # splitDates = row[0].split('/')
            # dates.append(date(year=int(splitDates[2]), month=int(splitDates[1]), day=int(splitDates[0])))
            dates.append(row[0])
            if(row[1] == '.'):
                rates.append(0.0)
            else:
                rates.append(row[1])
            line_count += 1

df = pd.DataFrame(list(zip(dates, rates)), columns=['Dates','Rates'])
print(df)
df['Dates'] = pd.to_datetime(df['Dates'])
df = df.set_index('Dates')
df['Rates'] = pd.to_numeric(df['Rates'])

print(df)
print(df.dtypes)

df.to_pickle('interestRates.pkl')