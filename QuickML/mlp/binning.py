import pandas as pd

class Binning:
    def __init__(self, data_frame):
        self._df=data_frame

    def binning_methods(self, table_data):

        for each in table_data:
            feature = each['feature']
            method = each['action']
            option = each['option']


            label = []
            binNum = 0

            #Manual Labeling
            if option == "Manual Labeling":
                label = each['label'].split(',')
                binNum = len(label)

            #Auto Labeling
            elif option == "Auto Labeling":
                binNum = each['binNumber']
                for i in range(1, binNum + 1):
                    auto_label = "Bin_" + str(i)
                    label.append(auto_label)

            #Binning By Width
            if method == 'Binning by Width':
                binned_items = pd.cut(self._df[feature], binNum, labels=label)
                self._df["Bin_" + feature] = binned_items

            #Binning By Frequency
            elif method == 'Binning by Frequency':
                binned_items = pd.qcut(self._df[feature], binNum, labels=label)
                self._df["Bin_" + feature] = binned_items

        return self._df
