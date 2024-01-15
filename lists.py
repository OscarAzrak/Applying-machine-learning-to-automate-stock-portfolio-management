import os
import pandas as pd

dates = ["2017-04-03","2018-04-03","2019-04-01","2020-04-01","2021-04-01","2022-04-01"]
years = ["2012", "2013", "2014","2015", "2016"]
total = []


seventeen = []
eighteen = []
nineteen = []
twenty = []
twentyone = []


for file in os.listdir('Fundamental_historik_inc_aktiedata'):
    company = []
    #fundamentals is a dataframe of the fundamentals sheet from the excel file for the company
    fundamentals = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "Year")
    # Set "Report" column as index
    fundamentals.set_index('Report', inplace=True)
    fundamentals = fundamentals[~fundamentals.index.duplicated(keep='first')]

    info = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "Info")
    #print(info["Unnamed: 2"][13])

    history = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "PriceDay")
    #Set the "Date" column as index
    history.set_index('Date', inplace=True)
    test = pd.date_range(start="2013-04-01", end="2022-04-13").difference(history.index)

    for i in range(0,len(dates)+1):
        year = str(i)

        try:
            if dates[i] not in test:
                if dates[i+1] not in test:
                    #print("Aktiehistorik finns för ", dates[i], "to", dates[i+1])
                    if fundamentals[years[i]]["Resultat Hänföring Aktieägare"]:
                        #print("Omsättning finns från", years[i])
                        if i == 0:
                            seventeen.append(file)
                        elif i == 1:
                            eighteen.append(file)
                        elif i == 2:
                            nineteen.append(file)
                        elif i == 3:
                            twenty.append(file)
                        elif i == 4:
                            twentyone.append(file)

        except (KeyError, IndexError) as e:
            pass

print(seventeen)
print(eighteen)
print(nineteen)
print(twenty)
print(twentyone)

