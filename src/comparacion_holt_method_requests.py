import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
import scipy
import sys


def normalize(data, interval):
	return data.resample(rule=str(interval) +'min').mean().bfill()

def test_bartlett(dataset1, dataset2):
	return scipy.stats.bartlett(dataset1, dataset2)

def test_welch(dataset1, dataset2):
	return scipy.stats.ttest_ind(dataset1, dataset2, equal_var = False)

def run():
	time_intervals = 5   #Duración de los intervalos en los que se dividen nuestros datos
	time_update = np.trunc(int(sys.argv[2]) / time_intervals)
	i = time_update

	fichero_leido = str(sys.argv[1])
	df = pd.read_csv(fichero_leido)

	df.timestamp = pd.to_datetime(df.timestamp, unit = 'ms')
	df['timestamp'] = df['timestamp'].dt.round('1s')

	df.index = df.timestamp
	df = normalize(df, time_intervals)

	new_length = int(np.trunc(len(df)/2))
	df_previo = df[:new_length+1]

	#df_previo.index = df_previo.timestamp
	#df_previo = normalize(df_previo)

	periods = len(df_previo)

	fit1 = ExponentialSmoothing(np.asarray(df_previo['requests']), seasonal_periods=int(periods), seasonal='add').fit()

	df_prediccion = df[new_length:]
	df_prediccion['Holt_Winter'] = fit1.forecast(len(df_prediccion))

	print(len(df))
	print(len(df_previo))
	for a in range(len(df_prediccion['Holt_Winter'])):
		if (df_prediccion['Holt_Winter'][a] < 0):
			df_prediccion['Holt_Winter'][a] = 0

	moving_average_predict = pd.Series.ewm(df_prediccion['Holt_Winter'], span = 10).mean()

	while i+time_update < len(df_previo):

		df_real = df[new_length:int(new_length+i)]
		print(len(df_real))
		print(str(new_length+i))


		#df_real.index = df_real.timestamp


		#df_real = normalize(df_real)

		#### Inicio del calculo de predicción
		#df_prediccion = df_real.copy()





		#df_prediccion['Holt_Winter'] = fit1.forecast(len(df_prediccion))





		plt.figure(figsize=(16,8))


		### Test de Bartlett para ver si hay diferencia significativa entre las desviaciones. Alpha es un valor de significancia que nosotros decidimos, por defecto sería 0.05
		alpha = 0.005

		bartlett_value, p_value_deviation = test_bartlett(df_prediccion['requests'], df_prediccion['Holt_Winter'])

		if p_value_deviation < alpha:
			print("\nTest de Bartlett fallido. La diferencia de desviación entre predicción y realidad es significativa\n")
			print (f'p-value de la desviacion es igual a: {p_value_deviation}\n')
		else:
			print("\nTest de Bartlett correcto. Desviación de consumo estable\n")


		### T-Test de Welch para ver si hay diferencia significativa entre las medias.
		welch_value, p_value_mean = test_welch(df_prediccion['requests'], df_prediccion['Holt_Winter'])

		if(df_real.mean()['requests'] >= df_prediccion.mean()['Holt_Winter'] and p_value_mean < alpha):
			veces = df_real.mean()['requests']/df_prediccion.mean()['Holt_Winter']
			print(f'Test de Welch fallido. La media de peticiones es {"%.2f" % veces} veces mayor que la predicción\n')
			print (f'p-value de la media es igual a: {p_value_mean}\n')
		elif (df_real.mean()['requests'] <= df_prediccion.mean()['Holt_Winter'] and p_value_mean < alpha):
			veces = df_real.mean()['Holt_Winter']/df_prediccion.mean()['requests']
			print(f'Test de Welch fallido. La media de peticiones es {"%.2f" % veces} veces menor que la predicción\n')
			print (f'p-value de la media es igual a: {p_value_mean}\n')
		else:
			print("Test de Welch correcto. Media de consumo estable\n")



		###Buscamos diferencias de 1500 peticiones o mas entre la predicción y la realidad, y comprobamos si se trata de un pico o es un cambio gradual
		moving_average_real = df_real['requests'].rolling(5).mean()

		plt.plot(df_previo['requests'], label='Nº requests previo')
		plt.plot(df_real['requests'], label='Nº requests real')
		plt.plot(moving_average_predict, label = 'Predicción de Nº requests')
		plt.plot(moving_average_real, label = 'Media móvil de datos reales')
		plt.legend(loc='best')
		for point in range(len(df_real['requests'])):
			if (moving_average_predict[point] - df_real['requests'][point] > 1000):
				plt.plot(df_real.index[point],df_real['requests'][point], 'ro', color = 'purple', markersize=8)

				if(len(moving_average_real) > 1 and moving_average_real[point-1] - df_real["requests"][point] > 800):
					print(f'ATENCIÓN. CAÍDA DE PETICIONES. SE PREDIJO {np.round(moving_average_predict[point])} PERO HA HABIDO {np.round(df_real["requests"][point])}')
					plt.plot(df_real.index[point],df_real['requests'][point], 'ro', color = 'red', markersize=8)


			elif (df_real['requests'][point] - moving_average_predict[point] > 1200):
				plt.plot(df_real.index[point],df_real['requests'][point], 'ro', color = 'purple', markersize=8)

				if(len(moving_average_real) > 1 and df_real['requests'][point] - moving_average_real[point-1] > 800):
					print(f'ATENCIÓN. SUBIDA DE PETICIONES. SE PREDIJO {np.round(moving_average_predict[point])} PERO HA HABIDO {np.round(df_real["requests"][point])}')
					plt.plot(df_real.index[point],df_real['requests'][point], 'ro', color = 'green', markersize=8)



		plt.show(block = True)
		plt.pause(3)
		plt.close()
		plt.pause(1)
		i+=time_update

if __name__ == "__main__":
	run()

	print("Fin de la ejecucion")
