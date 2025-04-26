import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('retail_store.csv')


x = data.drop(columns=['Demand Forecast'])
print(x.columns)
y = data['Demand Forecast']
print(y)

label_encoder_category = LabelEncoder()
label_encoder_region = LabelEncoder()
label_encoder_weather = LabelEncoder()
label_encoder_seasonality = LabelEncoder()
label_encoder_promotion = LabelEncoder()

x['Category'] = label_encoder_category.fit_transform(x['Category'])
x['Region'] = label_encoder_region.fit_transform(x['Region'])
x['Weather Condition'] = label_encoder_weather.fit_transform(x['Weather Condition'])
x['Seasonality'] = label_encoder_seasonality.fit_transform(x['Seasonality'])
x['Holiday/Promotion'] = label_encoder_promotion.fit_transform(x['Holiday/Promotion'])

scaler = StandardScaler()
x = scaler.fit_transform(x)

pickle.dump(scaler, open('scaler.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Train Score:", lr.score(X_train, y_train))
print("Test Score:", lr.score(X_test, y_test))

pickle.dump(lr, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

pickle.dump(label_encoder_category, open('Ctegory.pkl', 'wb'))
pickle.dump(label_encoder_region, open('Region.pkl', 'wb'))
pickle.dump(label_encoder_weather, open('weather.pkl', 'wb'))
pickle.dump(label_encoder_seasonality, open('season.pkl', 'wb'))
pickle.dump(label_encoder_promotion, open('promotion.pkl', 'wb'))

category_p = pickle.load(open('Ctegory.pkl', 'rb'))
region_p = pickle.load(open('Region.pkl', 'rb'))
weather_p = pickle.load(open('weather.pkl', 'rb'))
season_p = pickle.load(open('season.pkl', 'rb'))
promotion_p = pickle.load(open('promotion.pkl', 'rb'))

category_code = category_p.transform(['Toys'])[0]
region_code = region_p.transform(['South'])[0]
weather_code = weather_p.transform(['Sunny'])[0]
seasonality_code = season_p.transform(['Autumn'])[0]
promotion_code = promotion_p.transform([0])[0]

input_data = np.array([category_code, region_code, 204,
                       150, 66, 63.01, 20,
                       weather_code, promotion_code, 66.16,
                       seasonality_code]).reshape(1, -1)  # reshaped to 2D

final_features = scaler.transform(input_data)
prediction = model.predict(final_features)
print(prediction)