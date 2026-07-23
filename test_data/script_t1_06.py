from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_evaluate(X, y):
    # 1. تقسيم البيانات (80% تدريب و 20% اختبار)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. تطبيق Linear Regression
    model = LinearRegression()
    
    # 3. تدريب الموديل
    model.fit(X_train, y_train)
    
    # 4. التوقع y = mx + c
    y_pred = model.predict(X_test)
    
    # 5. حساب metrics (التقييم)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mae, r2

if __name__ == "__main__":
    from student1_data_preprocessing import get_data, preprocess_data
    
    # جلب وتجهيز البيانات باستخدام كود الطالب 1
    df = get_data()
    X, y = preprocess_data(df)
    
    # تدريب وتقييم الموديل
    model, X_test, y_test, y_pred, mae, r2 = train_and_evaluate(X, y)
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    print("\nModel Coefficients (Weights):")
    for col, coef in zip(X.columns, model.coef_):
        print(f"{col}: {coef:.2f}")