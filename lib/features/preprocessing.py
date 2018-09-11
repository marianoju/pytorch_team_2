import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocessing_housing():
    data = pd.read_csv("./data/housing.csv", na_values='')
    cleaned_data = data.copy()

    """ --------------------------------------------------------------------
    we fill missing values with mean values; 
    alternative is to drop all rows with NaN by 
    cleaned_data = .dropna(axis=0,how='any')
    -------------------------------------------------------------------- """
    cleaned_data = cleaned_data.fillna(data.mean())

    """ --------------------------------------------------------------------
    we convert a categorical variable into dummy/indicator variables 
    -------------------------------------------------------------------- """
    cleaned_data = pd.get_dummies(cleaned_data, drop_first=False)

    """ --------------------------------------------------------------------
    we define numerical and categorical (binary coded) variables 
    -------------------------------------------------------------------- """
    target = ["median_house_value"]
    numerical = ["longitude", "latitude", "housing_median_age",
                 "total_rooms", "total_bedrooms", "population",
                 "households", "median_income"]
    categorical = ["ocean_proximity_<1H OCEAN", "ocean_proximity_INLAND",
                   "ocean_proximity_ISLAND", "ocean_proximity_NEAR BAY",
                   "ocean_proximity_NEAR OCEAN"]

    scaler = StandardScaler()
    scaler.fit(cleaned_data[numerical])
    data_numerical = scaler.transform(cleaned_data[numerical])
    data_numerical = pd.DataFrame(data_numerical)
    data_numerical.columns = cleaned_data[numerical].columns

    """ --------------------------------------------------------------------
    we concatenate the rows with (scaled) numerical and categorical values 
    -------------------------------------------------------------------- """
    cleaned_data = pd.concat([data_numerical, cleaned_data[categorical],
                              data[target]], axis=1, sort=False)

    """ --------------------------------------------------------------------
    we define features and target
    -------------------------------------------------------------------- """
    data_X = cleaned_data.drop("median_house_value", axis=1)  # features
    data_y = cleaned_data["median_house_value"]  # target, or: dependent variable

    """ --------------------------------------------------------------------
    we split the dataset pseudo-randomly into training and testing subsets.
    -------------------------------------------------------------------- """
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,
                                                        test_size=0.2,
                                                        random_state=11)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    print("preprocessing() requires no input ")
    print("and returns as output: X_train, X_test, y_train, y_test")
