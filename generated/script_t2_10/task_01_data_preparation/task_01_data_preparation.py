from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # loading the data from csv file to pandas dataframe
    car_dataset = pd.read_csv('D:\PYTHON\ML\REGRESSION\cardata.csv')

    car_dataset.head()

    car_dataset.shape

    car_dataset.info()

    car_dataset.isnull().sum()

    print(car_dataset.Year.value_counts())
    print(car_dataset.Fuel_Type.value_counts())
    print(car_dataset.Seller_Type.value_counts())
    print(car_dataset.Transmission.value_counts())

    car_dataset.sample(4
                       )

    x_train, x_test, y_train, y_test = train_test_split(car_dataset.drop(['Car_Name', 'Selling_Price'],
                                                                         axis=1), car_dataset['Selling_Price'],
                                                                         test_size=0.2, random_state=2)

    categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
    transformer = ColumnTransformer(
        transformers=[
            ('tnf1', OrdinalEncoder(categories=[['Petrol','Diesel','CNG'],
                                                ['Dealer','Individual'],
                                                ['Manual','Automatic']]), categorical_cols),
            ('tnf2', MinMaxScaler(), ['Year', 'Present_Price', 'Kms_Driven'])
        ],
        remainder='passthrough'
    )

    transformer.fit(x_train)
    x_train = transformer.transform(x_train)
    x_test = transformer.transform(x_test)

    x_train.shape

    return transformer, x_test, x_train, y_test, y_train
