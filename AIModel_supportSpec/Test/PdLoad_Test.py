import pandas as pd
def loadData(filename):
    """
    Load data from the file
    """
    data = pd.read_csv(filename)
    return data

if __name__ == '__main__':
    filename = 'support_cases.csv'
    data = loadData(filename)
    print(data.head())  
    case_descriptions = data['case_description'].tolist()
    resolutions = data['resolution'].tolist()

    print (case_descriptions, resolutions)
