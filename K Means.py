# TAREA - K-MEANS con K=3, K=4 y K=5
# Luis P. Ferreras


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_theme(style='whitegrid', palette='muted')

# ---------------------------------------------------------
# 1. Cargar el csv
# ---------------------------------------------------------

# ajustar la ruta según donde tengas el archivo
perfil = pd.read_csv(r'perfil_clientes_con_clusters.csv')

print(perfil.shape)
print(perfil.head())

# las mismas variables que usamos en la clase de K-means
variables = [
    'n_transacciones',
    'monto_total',
    'monto_promedio',
    'monto_std',
    'n_canales',
    'n_tipos',
    'n_productos_prom',
    'prop_digital',
    'prop_exitosas'
]

X = perfil[variables]

# ---------------------------------------------------------
# 2. Escalar (con StandardScaler)
# ---------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 3. Probar K=3
# ---------------------------------------------------------

kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_3.fit(X_scaled)

perfil['cluster_k3'] = kmeans_3.labels_

print('Inercia K=3:', kmeans_3.inertia_)

# use otra métrica llamada
# "silhouette score" que sirve para comparar distintos K entre sí
# (mientras más cerca de 1, mejor separados quedan los grupos)
sil_3 = silhouette_score(X_scaled, kmeans_3.labels_)
print('Silhouette K=3:', sil_3)

print('\nEstadisticas por cluster, K=3:')
print(perfil.groupby('cluster_k3')[variables].mean().round(2))
print(perfil['cluster_k3'].value_counts())

# ---------------------------------------------------------
# 4. Probar K=4
# ---------------------------------------------------------

kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_4.fit(X_scaled)

perfil['cluster_k4'] = kmeans_4.labels_

print('\nInercia K=4:', kmeans_4.inertia_)
sil_4 = silhouette_score(X_scaled, kmeans_4.labels_)
print('Silhouette K=4:', sil_4)

print('\nEstadisticas por cluster, K=4:')
print(perfil.groupby('cluster_k4')[variables].mean().round(2))
print(perfil['cluster_k4'].value_counts())

# ---------------------------------------------------------
# 5. Probar K=5
# ---------------------------------------------------------

kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_5.fit(X_scaled)

perfil['cluster_k5'] = kmeans_5.labels_

print('\nInercia K=5:', kmeans_5.inertia_)
sil_5 = silhouette_score(X_scaled, kmeans_5.labels_)
print('Silhouette K=5:', sil_5)

print('\nEstadisticas por cluster, K=5:')
print(perfil.groupby('cluster_k5')[variables].mean().round(2))
print(perfil['cluster_k5'].value_counts())

# ---------------------------------------------------------
# 6. Comparar los 3 (pregunta 3 de la tarea)
# ---------------------------------------------------------

print('\n--- comparando K=3, K=4, K=5 ---')
print('K=3 -> inercia:', round(kmeans_3.inertia_, 1), ' silhouette:', round(sil_3, 4))
print('K=4 -> inercia:', round(kmeans_4.inertia_, 1), ' silhouette:', round(sil_4, 4))
print('K=5 -> inercia:', round(kmeans_5.inertia_, 1), ' silhouette:', round(sil_5, 4))

# la inercia siempre baja mientras más K le pongo, así que sola no sirve
# para decidir. Con el silhouette, K=3 me dio el numero mas alto (0.297)
# asi que me quedo con K=3 -> es donde los grupos se ven mas diferenciados

plt.figure(figsize=(8,4))
plt.bar(['K=3', 'K=4', 'K=5'], [sil_3, sil_4, sil_5], color=['#2196F3', '#FF5722', '#4CAF50'])
plt.title('Silhouette score por K')
plt.ylabel('silhouette')
plt.show()

# ---------------------------------------------------------
# 7. Nombres de negocio para K=3 (pregunta 4)
# ---------------------------------------------------------

# reviso perfil.groupby('cluster_k3')[variables].mean() de arriba y
# le pongo nombre a cada cluster según lo que más se destaca

nombres = {
    0: 'Clientes normales',
    1: 'Clientes VIP',
    2: 'Clientes con varios productos'
}

perfil['segmento'] = perfil['cluster_k3'].map(nombres)

print('\nSegmentos con K=3:')
print(perfil.groupby('segmento')[variables].mean().round(2))
print(perfil['segmento'].value_counts())

# grafico para ver los clusters de K=3
colores = {0:'#2196F3', 1:'#FF5722', 2:'#4CAF50'}
plt.figure(figsize=(8,5))
for c in perfil['cluster_k3'].unique():
    sub = perfil[perfil['cluster_k3'] == c]
    plt.scatter(sub['n_transacciones'], sub['monto_promedio'],
                alpha=0.5, label=nombres[c], color=colores[c])
plt.xlabel('n_transacciones')
plt.ylabel('monto_promedio')
plt.legend()
plt.title('Clusters con K=3')
plt.show()

# ---------------------------------------------------------
# 8. Cuál cluster necesita mas atencion (pregunta 5)
# ---------------------------------------------------------

# para pensarlo mejor, calculo qué % del monto total le toca a cada
# segmento (no solo cuántos clientes tiene cada uno)

monto_por_segmento = perfil.groupby('segmento')['monto_total'].sum()
porcentaje = monto_por_segmento / monto_por_segmento.sum() * 100

print('\n% del monto total por segmento:')
print(porcentaje.round(1))

# con esto veo que "Clientes con varios productos" (que son 125) suman
# casi lo mismo o un poco mas de monto total que "Clientes VIP" (que
# son solo 36). Osea que aunque el VIP tiene el monto mas alto POR
# cliente, el otro grupo pesa parecido en total porque son mas.
# Todavia no se cual es "la" respuesta correcta, pero mi primera idea
# es que el VIP es mas urgente porque son pocos y si se va uno se nota
# mucho, y el de "varios productos" es una oportunidad de crecer porque
# es un grupo mas grande y usan poco lo digital todavia.

# guardo el resultado final
perfil.to_csv('perfil_con_segmentos_k3.csv', index=False)
print('\nListo, guardado perfil_con_segmentos_k3.csv')
