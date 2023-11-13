def multiVariateRegressionModel(df, features, target):
    features = [col for col in df.columns if col != 'target']
    X = df[features]
    y = df[target]

    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    
