# librerias que nuestro profesor nos recomendó para poder trabajar con el algoritmo PSO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps


# Aquí lo primero que hacemos es cargar los datos de nuestro archivo csv, esto lo hacemos para poder guardar las variables 
# como longitud, latitud, etc, y poder trabajar con ellas en el algoritmo de enjambre.
datos = pd.read_csv("datos.csv") # cargamos los datos del archivo csv "datos.csv"

# Mostramos las primeras filas de nuestro archivo de excel 
print("Primeras filas de los datos:\n")
print(datos.head())

# Aquí lo que hacemos es extraer las coordenadas y cultivos.
latitudes = datos['Latitud'].values
longitudes = datos['Longitud'].values
cultivos = datos['Cultivo'].values


# Parte fundamental del algoritmo: Función objetivo con factores de cultivo normalizados
# aquí lo que haremos es evaluar que tan buena es la posible solución, que en este caso, 
# esa posible solución sería una posible ubicación de algún sensor que se considere bueno en el campo agrícola

def funcion_objetivo(posiciones):
    num_particulas = posiciones.shape[0]
    costos = np.zeros(num_particulas)
    
    # Valores que tomaremos para normalizar, esto lo haremos con el fin de que todos los factores 
    # tengan un impacto comparable cuando lo combinemos con los pesos.
    humedades = datos['Humedad'].values
    elevaciones = datos['Elevacion'].values
    salinidades = datos['Salinidad'].values
    temperaturas = datos['Temperatura'].values

    hum_norm = (100 - humedades) / 100
    elev_norm = elevaciones / elevaciones.max()
    sal_norm = salinidades / salinidades.max()
    temp_norm = abs(temperaturas - 30) / 30  

    # Estos son los pesos de nuestra funcion fitnes, y son ajustables dependiendo de que objetivo 
    # queremos lograr.
    peso_hum, peso_elev, peso_sal, peso_temp = 0.4, 0.2, 0.2, 0.2

    # aqui combinamos los factores normalizados, con el fin de obtener un solo valor por cultivo que mide lo dificil o 
    # en otras palabras, lo costoso que puede salir cubrir ese cultivo
    factor_total = (
        peso_hum * hum_norm
        + peso_elev * elev_norm
        + peso_sal * sal_norm
        + peso_temp * temp_norm
    )
   
# aqui iiteramos sobre cada particula, y cada particula representa una posible distribución de todos los sensores
    for i in range(num_particulas):
        posicion_sensores = posiciones[i].reshape(-1, 2)
        costo_total = 0
        
        # aquí calculamos la distancia mínima a cada cultivo, si el cultivo está en malas condiciones, 
        # el costo aumenta.
        for lat_cultivo, lon_cultivo, factor in zip(latitudes, longitudes, factor_total):
            distancia_minima = np.min(
                np.sqrt((posicion_sensores[:, 0] - lat_cultivo) ** 2 + (posicion_sensores[:, 1] - lon_cultivo) ** 2)
            )
            costo_total += distancia_minima * factor
        
        costos[i] = costo_total

    return costos


# Aquí configurams el PSO
num_sensores = 10 # numero de sensores con los que trabajaremos
num_particulas = 70 # Número de particulas que usarmeos 

lat_min, lat_max = latitudes.min(), latitudes.max()
lon_min, lon_max = longitudes.min(), longitudes.max()

limites = (
    np.array([lat_min, lon_min] * num_sensores),
    np.array([lat_max, lon_max] * num_sensores)
)

optimizador = ps.single.GlobalBestPSO(
    n_particles=num_particulas,
    dimensions=2 * num_sensores,
    options={'c1': 0.8, 'c2': 0.5, 'w': 0.7},
    bounds=limites
)


# Aquí ejecutaremos la optimización, veremos cual es el mejor valor encontrado (mejor partícula) e imprimiremos 
# la longitud y latitud del mejor sensor para tales cultivos

mejor_costo, mejor_posicion = optimizador.optimize(funcion_objetivo, iters=150)

sensores_lat = mejor_posicion[::2]
sensores_lon = mejor_posicion[1::2]

print("\nMejor valor encontrado:", mejor_costo)
print("Mejor configuración de sensores (latitud, longitud):")
for i, (lat, lon) in enumerate(zip(sensores_lat, sensores_lon), start=1):
    print(f"Sensor {i}: Latitud {lat:.6f}, Longitud {lon:.6f}")


# Aquí, por medio de matplot, graficaremos visualmente con cobertura real la ubicacion de cada sensor 

radio_cobertura = 0.01 # lo podemos ajustar para ver la cobertura de cada sensor

def cultivos_cubiertos(sensor_lat, sensor_lon, radio):
    return np.sqrt((latitudes - sensor_lat)**2 + (longitudes - sensor_lon)**2) <= radio

plt.figure(figsize=(10, 8))
colores = {'Maíz': 'green', 'Chile': 'red', 'Tomate': 'orange'}

# Graficamos los cultivos 
for cultivo in colores.keys():
    indices = cultivos == cultivo
    plt.scatter(longitudes[indices], latitudes[indices], c=colores[cultivo], label=cultivo, alpha=0.6)

# Graficamos los sensores y su cobertura propia.
cultivos_cubiertos_total = np.zeros(len(latitudes), dtype=bool)
for i, (lat, lon) in enumerate(zip(sensores_lat, sensores_lon), start=1):
    plt.scatter(lon, lat, c='blue', marker='X', s=150)
    circulo = plt.Circle((lon, lat), radio_cobertura, color='blue', fill=True, alpha=0.2)
    plt.gca().add_patch(circulo)
    plt.text(lon, lat + 0.002, f'Sensor {i}', color='blue', fontsize=9, fontweight='bold')
    
    cubiertos = cultivos_cubiertos(lat, lon, radio_cobertura)
    cultivos_cubiertos_total = cultivos_cubiertos_total | cubiertos

print(f"\nCultivos cubiertos por al menos un sensor: {cultivos_cubiertos_total.sum()} de {len(latitudes)}")

plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.title('Ubicación de Cultivos y Sensores con Cobertura')
plt.grid(True)
plt.legend()
plt.show()


# Para terminar, evaluamos la distribución de los sensores, para ver que tan efectivo fue cada distribución 

distancias_minimas = np.zeros(len(latitudes))
for i, (lat_cultivo, lon_cultivo) in enumerate(zip(latitudes, longitudes)):
    distancias = np.sqrt((sensores_lat - lat_cultivo)**2 + (sensores_lon - lon_cultivo)**2)
    distancias_minimas[i] = distancias.min()

distancia_promedio = distancias_minimas.mean()
print(f"\nDistancia promedio de cada cultivo al sensor más cercano: {distancia_promedio:.6f}")

cultivos_en_rango = distancias_minimas <= radio_cobertura
porcentaje_cubiertos = np.sum(cultivos_en_rango) / len(latitudes) * 100
print(f"Porcentaje de cultivos cubiertos por al menos un sensor: {porcentaje_cubiertos:.2f}%")
