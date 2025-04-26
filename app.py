import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
category_p = pickle.load(open('Ctegory.pkl', 'rb'))
region_p = pickle.load(open('Region.pkl', 'rb'))
weather_p = pickle.load(open('weather.pkl', 'rb'))
season_p = pickle.load(open('season.pkl', 'rb'))
promotion_p = pickle.load(open('promotion.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    category = request.form['category']
    region = request.form['region']
    inventory = int(request.form['inventory'])
    units_sold = int(request.form['units_sold'])
    units_order = int(request.form['units_order'])
    price = float(request.form['price'])
    discount = int(request.form['discount'])
    competitor = float(request.form['competitor'])
    weather = request.form['weather_condition']
    season = request.form['seasonality']
    promotion = int(request.form['holiday_promotion'])

    category_code = category_p.transform([category])[0]
    region_code = region_p.transform([region])[0]
    weather_code = weather_p.transform([weather])[0]
    seasonality_code = season_p.transform([season])[0]
    promotion_code = promotion_p.transform([promotion])[0]

    input_data = np.array([category_code, region_code, inventory, units_sold,
                           units_order, price, discount, weather_code,
                           promotion_code, competitor, seasonality_code])
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    final_features = scaler.transform([input_data])

    prediction = model.predict(final_features)
    output = prediction
    return render_template('index.html', prediction_text='''Remand Forecast
                            Should be {}'''.format(output))


if __name__ == '__main__':
    app.run(debug=True)
