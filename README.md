# SolarSync-Predictor

* Problem: Predict hourly solar or wind energy output to optimize grid stability and energy storage.
* Dataset: Use the Solar Power Generation dataset [Link](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data?resource=download)
### Approach:
* Engineer time-series features like historical output, weather conditions, and time of day.
* Frame as a regression problem; train models like Gradient Boosting, LSTM, and a transformer-based time-series model.
* Use time-series cross-validation to avoid look-ahead bias.
* Deploy a Dash dashboard integrated with a weather API for real-time energy forecasts.
[Live Demo](https://huggingface.co/spaces/mahmad92/SolarSyncPredictor)

![Perview01](https://github.com/Muhammad-Ahmad092/SolarSync-Predictor/blob/main/perview%2001.png)
![Perview02](https://github.com/Muhammad-Ahmad092/SolarSync-Predictor/blob/main/perview%2002.png)
![Perview03](https://github.com/Muhammad-Ahmad092/SolarSync-Predictor/blob/main/perview%2003.png)
