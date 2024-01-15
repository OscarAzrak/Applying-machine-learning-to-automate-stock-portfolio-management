import os
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.metrics import classification_report

from classifiers import *
import matplotlib.pyplot as plt

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 1500)


def main():

    training_trade_dates = ["2017-04-03","2018-04-03","2019-04-01","2020-04-01","2021-04-01"]
    testing_trading_dates = ["2021-04-01","2022-04-01"]

    training_periods = ["2012", "2013", "2014","2015", "2016",'2017','2018','2019', '2020']
    testing_periods = ['2017','2021']

    active = True
    while active:
        choice = menu()
        if choice == '0':
            print('You chose to exit the program. Goodbye!\n')
            quit()
        elif choice == "1":
            print("Creating training data")
            training_lists = create_training_lists(training_periods, training_trade_dates)
            create_training_pickle(training_lists)
            """Load training Pickle"""
        elif choice == "2":
            print("Creating testing data")
            x_test = test_matrices(testing_trading_dates, testing_periods)
            create_testing_pickle(x_test)
            """Load testing pickle"""
        elif choice == "3":
            print("Creating portfolios from classifiers")

            X_train, Y_train = read_text_pickles()
            x_test, Y_test = read_pickle_test()

            LogReg(X_train, Y_train, x_test)
            MLP(X_train, Y_train, x_test)
            RF(X_train, Y_train, x_test)
            KNN(X_train, Y_train, x_test)
            extreme_gradient_boosting(X_train, Y_train, x_test)
        elif choice == "4":
            print("Printing portfolios for each classifier: ")

            print()
            print("LOG return: ")
            get_portfolio("X_testLog")
            print()

            print()
            print("RF return: ")
            get_portfolio("X_test_RF")
            print()

            print()
            print("KNN return: ")
            get_portfolio("X_test_KNN")
            print()

            print()
            print("EGB return: ")
            get_portfolio("X_test_EGB")
            print()

            print()
            print("MLP return: ")
            get_portfolio("X_testMLP")
            print()

        elif choice == "5":
            print("Create returns of each portfolios")
            get_portfolio_return("X_testLog", "Log")
            get_portfolio_return("X_test_RF", "RF")
            get_portfolio_return("X_test_KNN", "KNN")
            get_portfolio_return("X_test_EGB", "EGB")
            get_portfolio_return("X_testMLP", "MLP")
            index_return()


        elif choice == "6":
            print("Plotting results and index")
            plot()






def create_training_lists(training_periods, training_trade_dates):
    training_lists = []

    for file in os.listdir('Fundamental_historik_inc_aktiedata'):
        """ fundamentals is a dataframe of the fundamentals sheet from the excel file for the company """
        fundamentals = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "Year", engine='openpyxl')
        # Set "Report" column as index
        fundamentals.set_index('Report', inplace=True)
        fundamentals = fundamentals[~fundamentals.index.duplicated(keep='first')]

        info = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "Info", engine='openpyxl')
        info.set_index("Unnamed: 1", inplace=True)



        """ history is a dataframe of the historic price for the stock form the excel file """
        history = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "PriceDay", engine='openpyxl')
        #Set the "Date" column as index
        history.set_index('Date', inplace=True)

        """dates is a dataframe that contains dates from 2013-04-01 to 2022-04-13 that history dataframe does NOT have"""
        dates = pd.date_range(start="2013-04-01", end="2022-04-13").difference(history.index)


        """ Get all companies with at least 750MSEK as marketcap (if marketcap is 600MSEK 2012 and 770MSEK 2013, company is included) """
        try:
            if (fundamentals.iloc[:,-1]["Antal Aktier"] * fundamentals.iloc[:,-1]["Aktiekurs Snitt"]) >= 750:
                """IF market cap above 750MSEK: continue for loop with current stock"""
                pass
            else:
                """IF market cap BELOW 750MSEK: continue: this will check next stock's marketcap"""
                continue  # only executed if the inner loop did NOT break

        except KeyError:
            """If KeyError is raised: check next stock"""
            continue

        #Checking 5 year periods only
        # if wanting to buy stock in year i, we need fundamentals for years i-5 to i-1, example: wanting to buy stocks in april 2017, we need fundamentals to train on for years 2012 to 2016.
        for i in range(0,4):
            company = []
            try:
                #if the trading date is NOT in dates dataframe, that means that the stock has traded that day
                if training_trade_dates[i] not in dates:

                    deltarevenue = round((fundamentals[training_periods[i+4]]["Nettoomsättning"]/fundamentals[training_periods[i]]["Nettoomsättning"])-1, 3)
                    company.append(deltarevenue)

                    deltaresult = round((fundamentals[training_periods[i+4]]["Resultat Hänföring Aktieägare"]/fundamentals[training_periods[i]]["Resultat Hänföring Aktieägare"])-1, 3)
                    company.append(deltaresult)

                    deltafcf = round((fundamentals[training_periods[i+4]]["FrittKassaflöde"]/fundamentals[training_periods[i]]["FrittKassaflöde"])-1,3)
                    company.append(deltafcf)

                    netincome = (fundamentals[training_periods[i+4]]["Resultat Hänföring Aktieägare"])
                    dividends = (fundamentals[training_periods[i+4]]["Antal Aktier"]) * (fundamentals[training_periods[i+4]]["Utdelning"])
                    equity_debt = (fundamentals[training_periods[i+4]]["Totala Skulder och Eget kapital"])

                    roic = round((netincome-dividends)/(equity_debt), 3)
                    company.append(roic)

                    debtequity = round((fundamentals[training_periods[i+4]]["Totala Skulder"])/fundamentals[training_periods[i+4]]["Summa Eget Kapital"], 3)
                    company.append(debtequity)

                    selling_close = history.loc[training_trade_dates[i+1]]['Closeprice'].values[0]
                    buying_close = history.loc[training_trade_dates[i]]['Closeprice'].values[0]
                    YoY = round((selling_close/buying_close) - 1,3)
                    company.append(YoY)

                    name = info["Unnamed: 2"]["Bolag"]
                    company.append(name)
                    training_lists.append(company)

            except (KeyError, IndexError):
                pass

    return training_lists

#save training_lists as pickle so there's no need to train same files multiple times.
def create_training_pickle(training_lists):
    x_matrix = pd.DataFrame(training_lists)
    x_matrix.columns = ['Revenue', "Result", "FCF", "ROIC", "D/E", "YoY", "Name"]

    x_matrix.sort_values(by=['YoY'], ascending=False, inplace=True)

    x_matrix.set_index('Name', inplace=True)
    x_matrix.to_pickle('x_matrix.pickle')



#clean pickle file and create y_vector
def read_text_pickles():
    x_matrix = pd.read_pickle('x_matrix.pickle')
    x_matrix = x_matrix.fillna(0)
    x_matrix = x_matrix.replace([np.inf, -np.inf], 0).dropna(how="all")
    print(len(x_matrix.index))
    Y_train = []
    for i in x_matrix["YoY"]:
        if round(i,2) >= 0.25:
            Y_train.append(1)
        else:
            Y_train.append(0)

    x_matrix = x_matrix.drop(columns=['YoY'], axis=1)

    return x_matrix, Y_train

#create test lists, similar as training
def test_matrices(testing_trading_dates, testing_periods):

    test_matrix = []

    for file in os.listdir('Fundamental_historik_inc_aktiedata'):
        """ fundamentals is a dataframe of the fundamentals sheet from the excel file for the company """
        fundamentals = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "Year", engine='openpyxl')
        # Set "Report" column as index
        fundamentals.set_index('Report', inplace=True)
        fundamentals = fundamentals[~fundamentals.index.duplicated(keep='first')]

        info = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "Info", engine='openpyxl')
        info.set_index("Unnamed: 1", inplace=True)


        """ history is a dataframe of the historic price for the stock form the excel file """
        history = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "PriceDay", engine='openpyxl')
        #Set the "Date" column as index
        history.set_index('Date', inplace=True)

        dates = pd.date_range(start="2013-04-01", end="2022-04-13").difference(history.index)
        """Get the companies that have a market cap of 750MSEK"""
        try:
            if (fundamentals.iloc[:,-1]["Antal Aktier"] * fundamentals.iloc[:,-1]["Aktiekurs Snitt"]) >= 750:
                """IF market cap above 750MSEK: continue for loop with current stock"""
                #print(file, fundamentals[all_periods[i]]["Antal Aktier"] * fundamentals[all_periods[i]]["Aktiekurs Snitt"])
                pass
            else:
                """IF market cap BELOW 750MSEK: continue: this will check next stock's marketcap"""
                continue  # only executed if the inner loop did NOT break

        except KeyError:
            """If KeyError is raised: check next stock"""
            continue

        company = []

        try:
            if testing_trading_dates[0] not in dates:
                deltarevenue = round((fundamentals[testing_periods[1]]["Nettoomsättning"]/fundamentals[testing_periods[0]]["Nettoomsättning"])-1, 3)
                company.append(deltarevenue)

                deltaresult = round((fundamentals[testing_periods[1]]["Resultat Hänföring Aktieägare"]/fundamentals[testing_periods[0]]["Resultat Hänföring Aktieägare"])-1, 3)
                company.append(deltaresult)

                deltafcf = round((fundamentals[testing_periods[1]]["FrittKassaflöde"] / fundamentals[testing_periods[0]]["FrittKassaflöde"]) - 1, 3)
                company.append(deltafcf)


                netincome = (fundamentals[testing_periods[1]]["Resultat Hänföring Aktieägare"])
                dividends = (fundamentals[testing_periods[1]]["Antal Aktier"]) * (fundamentals[testing_periods[1]]["Utdelning"])
                equity_debt = (fundamentals[testing_periods[1]]["Totala Skulder och Eget kapital"])
                roic = round((netincome-dividends)/(equity_debt),3)
                company.append(roic)



                debtequity = round((fundamentals[testing_periods[1]]["Totala Skulder"]) / fundamentals[testing_periods[1]]["Summa Eget Kapital"], 3)
                company.append(debtequity)


                sell_close = history.loc[testing_trading_dates[1]]['Closeprice'].values[0]
                buy_close = history.loc[testing_trading_dates[0]]['Closeprice'].values[0]


                YoY = round((sell_close/buy_close) - 1,3)
                company.append(YoY)

                name = info["Unnamed: 2"]["Bolag"]
                company.append(name)
                test_matrix.append(company)



        except (KeyError, IndexError):
            pass
    return test_matrix

def create_testing_pickle(x_matrix):
    x_matrix = pd.DataFrame(x_matrix)
    x_matrix.columns = ['Revenue', "Result", "FCF", "ROIC", "D/E", "YoY", "Name"]

    x_matrix.set_index('Name', inplace=True)
    x_matrix.to_pickle('X_test.pickle')


def read_pickle_test():
    x_test = pd.read_pickle('X_test.pickle')
    x_test = x_test.fillna(0)
    x_test = x_test.replace([np.inf, -np.inf], 0).dropna(how="all")
    print(len(x_test.index))
    Y_True = []
    for i in x_test["YoY"]:
        if round(i,2) >= 0.25:
            Y_True.append(1)
        else:
            Y_True.append(0)

    x_test = x_test.drop(columns=['YoY'], axis=1)


    return x_test, Y_True




def get_portfolio(method):
    pickling = method + ".pickle"

    X = pd.read_pickle(pickling)
    X.sort_values(by=["Y_test_Probs"], ascending=False, inplace=True)
    top25 = X.head(25)

    X = X[X.iloc[:, -2] == 1]

    print("All stocks: ",len(X.index))
    for name in X.index:
        print(name, end=', ')
    print()

    print("Top 25 stocks: ",len(top25.index))
    for name in top25.index:
        print(name, end=', ')


def get_portfolio_return(method, methodtopickle):
    pickling = method + ".pickle"

    X = pd.read_pickle(pickling)
    X.sort_values(by=["Y_test_Probs"], ascending=False, inplace=True)
    top25 = X.head(25)

    X = X[X.iloc[:, -2] == 1]

    portfolio_close_ones = pd.DataFrame()
    portfolio_close_top25 = pd.DataFrame()

    weight_ones = []
    weight_top25 = []
    for i in range(len(X.index)):
        weight_ones.append(1/(len(X.index)))

    for i in range(len(top25.index)):
        weight_top25.append(1/(len(top25.index)))

    for file in os.listdir('Fundamental_historik_inc_aktiedata'):
        info = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "Info", engine='openpyxl')
        info.set_index("Unnamed: 1", inplace=True)
        name = info["Unnamed: 2"]["Bolag"]

        if name in top25.index:
            """ history is a dataframe of the historic price for the stock form the excel file """
            history = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "PriceDay", engine='openpyxl')
            #Set the "Date" column as index
            history.set_index('Date', inplace=True)
            portfolio_close_top25[name] = history["Closeprice"]
        if name in X.index:
            """ history is a dataframe of the historic price for the stock form the excel file """
            history = pd.read_excel('Fundamental_historik_inc_aktiedata/{}'.format(file), "PriceDay", engine='openpyxl')
            #Set the "Date" column as index
            history.set_index('Date', inplace=True)
            portfolio_close_ones[name] = history["Closeprice"]
        else:
            continue

    """ALL stocks part"""
    mask = (portfolio_close_ones.index >= '2021-4-1') & (portfolio_close_ones.index <= '2022-4-1')
    relevant_dates_ones = portfolio_close_ones.loc[mask]
    relevant_dates_ones = relevant_dates_ones.reindex(index=relevant_dates_ones.index[::-1])

    daily_pct_change_ones = relevant_dates_ones.pct_change()
    daily_pct_change_ones.fillna(0, inplace=True)
    weighted_return_ones = (weight_ones * daily_pct_change_ones)
    portfolio_return_ones = weighted_return_ones.sum(axis=1)
    cumprod_portfolio_ones = (1 + portfolio_return_ones).cumprod()


    pickling_all = methodtopickle + "_return_all.pickle"
    cumprod_portfolio_ones.to_pickle(pickling_all)

    """Top 30 part"""
    mask1 = (portfolio_close_top25.index >= '2021-4-1') & (portfolio_close_top25.index <= '2022-4-1')

    relevant_dates_top25 = portfolio_close_top25.loc[mask1]
    relevant_dates_top25 = relevant_dates_top25.reindex(index=relevant_dates_top25.index[::-1])

    daily_pct_change_top25 = relevant_dates_top25.pct_change()
    daily_pct_change_top25.fillna(0, inplace=True)
    weighted_return_top25 = (weight_top25 * daily_pct_change_top25)
    portfolio_return_top25 = weighted_return_top25.sum(axis=1)
    cumprod_portfolio_top25 = (1 + portfolio_return_top25).cumprod()


    pickling_all = methodtopickle + "_return_top25.pickle"
    cumprod_portfolio_top25.to_pickle(pickling_all)

def index_return():
    index = pd.read_excel("OMXS30-Omx Sthlm 30.xlsx", "PriceDay")
    index.set_index('Date', inplace=True)
    mask = (index.index >= '2021-4-1') & (index.index <= '2022-4-1')
    relevant_dates = index.loc[mask]
    #reverse order
    relevant_dates = relevant_dates.reindex(index=relevant_dates.index[::-1])


    relevant_dates = relevant_dates["Closeprice"]
    daily_pct_change = relevant_dates.pct_change()
    daily_pct_change.fillna(0, inplace=True)
    cumprod_daily_pct_change = (1 + daily_pct_change).cumprod()
    index = cumprod_daily_pct_change
    index.to_pickle('index_cum_return.pickle')





def plot():
    x_test, Y_test = read_pickle_test()

    ret_EGB_all = pd.read_pickle('EGB_return_all.pickle')
    ret_EGB_top25 = pd.read_pickle('EGB_return_top25.pickle')
    EGB_report = pd.read_pickle('X_test_EGB.pickle').iloc[:,-2].tolist()


    ret_RF_all = pd.read_pickle('RF_return_all.pickle')
    ret_RF_top25 = pd.read_pickle('RF_return_top25.pickle')
    RF_report = pd.read_pickle('X_test_RF.pickle').iloc[:,-2].tolist()

    ret_KNN_all = pd.read_pickle('KNN_return_all.pickle')
    ret_KNN_top25 = pd.read_pickle('KNN_return_top25.pickle')
    KNN_report = pd.read_pickle('X_test_KNN.pickle').iloc[:,-2].tolist()

    ret_LOG_all = pd.read_pickle('Log_return_all.pickle')
    ret_LOG_top25 = pd.read_pickle('Log_return_top25.pickle')
    LOG_report = pd.read_pickle('X_testLog.pickle').iloc[:,-2].tolist()

    ret_MLP_all = pd.read_pickle('MLP_return_all.pickle')
    ret_MLP_top25 = pd.read_pickle('MLP_return_top25.pickle')
    MLP_report = pd.read_pickle('X_testMLP.pickle').iloc[:,-2].tolist()

    index = pd.read_pickle("index_cum_return.pickle")

    print("Return of Extreme Gradient Boosting Full Portfolio: " , (round(ret_EGB_all.iloc[-1]-1, 3))*100, "%")
    print("Return of Extreme Gradient Boosting Top 25 Portfolio: " , (round(ret_EGB_top25.iloc[-1]-1, 3))*100, "%")
    print(classification_report(Y_test, EGB_report, target_names=['IGNORE', 'BUY']))
    print()

    print("Return of Random Forest Full Portfolio: " , (round(ret_RF_all.iloc[-1]-1, 3))*100, "%")
    print("Return of Random Forest Top 25 Portfolio: " , (round(ret_RF_top25.iloc[-1]-1, 3))*100, "%")
    print(classification_report(Y_test, RF_report, target_names=['IGNORE', 'BUY']))
    print()

    print("Return of K Nearest Neighbour Full Portfolio: " , (round(ret_KNN_all.iloc[-1]-1, 3))*100, "%")
    print("Return of K Nearest Neighbour Top 25 Portfolio: " , (round(ret_KNN_top25.iloc[-1]-1, 3))*100, "%")
    print(classification_report(Y_test, KNN_report, target_names=['IGNORE', 'BUY']))
    print()
    print("Return of Logistic Regression Full Portfolio: " , (round(ret_LOG_all.iloc[-1]-1, 3))*100, "%")
    print("Return of Logistic Regression Top 25 Portfolio: " , (round(ret_LOG_top25.iloc[-1]-1, 3))*100, "%")
    print(classification_report(Y_test, LOG_report, target_names=['IGNORE', 'BUY']))
    print()

    print("Return of MLP Full Portfolio: " , (round(ret_MLP_all.iloc[-1]-1, 3))*100, "%")
    print("Return of MLP Top 25 Portfolio: " , (round(ret_MLP_top25.iloc[-1]-1, 3))*100, "%")
    print(classification_report(Y_test, MLP_report, target_names=['IGNORE', 'BUY']))
    print()

    print("Return of index Portfolio: " , (round(index.iloc[-1]-1, 3))*100, "%")


    merged = pd.concat([ret_EGB_all, ret_RF_all, ret_KNN_all,ret_LOG_all,ret_MLP_all, index], axis=1, join="inner")
    merged.columns = ["EGB", "RF", "KNN","LogReg", "MLP", "Index"]

    fig, axs = plt.subplots(2)
    fig.suptitle('Full Portfolio (Top) VS Top 25 Portfolio (Bottom)')
    #axs[0].plot(x, y)
    #axs[1].plot(x, -y)


    #plt.figure(figsize=(15, 7))

    for stock in merged:
        axs[0].plot(merged[stock], label=stock)
    axs[0].legend()
    #axs[0].ylabel("Returns")
    #axs[0].xlabel("Date")

    ##Top 30 plot
    merged2 = pd.concat([ret_EGB_top25, ret_RF_top25, ret_KNN_top25,ret_LOG_top25,ret_MLP_top25, index], axis=1, join="inner")
    merged2.columns = ["EGB", "RF", "KNN", "LogReg" ,"MLP", "Index"]
    #plt.figure(figsize=(15, 7))
    for stock in merged2:
        axs[1].plot(merged2[stock], label=stock)
    axs[1].legend()
    #axs[1].ylabel("Returns")
    #axs[1].xlabel("Date")
    plt.show()

def menu():
    options = ["1", "2", "3", "4", "5","6","0", "q"]

    print('' +
          '\n0. Quit.' +
          '\n1. Create training matrices.' +
          '\n2. Create testing matrices.' +
          '\n3. Create portfolios from each classifier.' +
          '\n4. Print portfolios for each classifier.' +
          '\n5. Create returns of each portfolios.' +
          '\n6. Plot return of each portfolio')
    choice = input('Select an option from the menu and press enter:\t')
    choice = choice.strip()
    choice = choice.lower()

    if choice == 'q' or choice == 'quit':
        print('You quit the program!\n')
        quit()
    while choice not in options:
        choice = input('Invalid input, try again: ')
        choice = choice.strip()
        choice = choice.lower()

        if choice == 'q' or choice == 'quit':
            print('You chose to exit the program. Goodbye!\n')
            quit()
    return choice

main()