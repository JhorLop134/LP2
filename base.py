import pandas as pd
import numpy as np

class EstadisticaBase:
    """
    Clase base abstracta para todos los tipos de análisis estadístico.
    Define la estructura básica, validaciones y encapsulamiento.
    """
    def __init__(self, datos):
        """
        Constructor que recibe los datos y los normaliza a una Serie de Pandas.
        """
        if datos is None or (isinstance(datos, (list, tuple)) and not datos):
            raise ValueError("La lista de datos no puede estar vacía.")
            
        # Normalizar datos a pd.Series
        if isinstance(datos, (list, tuple)):
            self.datos = pd.Series(datos)
        elif isinstance(datos, pd.Series):
            self.datos = datos
        else:
            raise TypeError("El formato de datos debe ser lista, tupla o pd.Series.")

        # Atributo de encapsulamiento
        self._n_observaciones = len(self.datos.dropna())

    # Método de Polimorfismo (Será sobrescrito en cualitativos.py y cuantitativos.py)
    def resumen(self):
        """
        Método que proporciona un resumen de las métricas clave.
        Debe ser implementado por las clases hijas.
        """
        raise NotImplementedError("Las clases hijas deben implementar el método resumen()")
    
    # Método para Encapsulamiento (Acceso controlado a los datos)
    def obtener_datos(self):
        """Devuelve los datos de la Serie de Pandas."""
        return self.datos

    def obtener_n_observaciones(self):
        """Devuelve el número de observaciones no nulas."""
        return self._n_observaciones

# Ejemplo de uso (opcional, para prueba):
# datos_prueba = [1, 2, 3, 4, 5]
# base = EstadisticaBase(datos_prueba)
# print(base.obtener_n_observaciones())

    def contar_datos(self):
        """Devuelve la cantidad de elementos en el conjunto de datos."""
        return len(self.datos)

    def suma(self):
        """Calcula la suma total de los valores."""
        return sum(self.datos)

    def media(self):
        """Calcula la media aritmética sin funciones externas."""
        n = self.contar_datos()
        return self.suma() / n if n > 0 else float('nan')

    def mediana(self):
        """Calcula la mediana ordenando los datos manualmente."""
        n = self.contar_datos()
        if n == 0:
            return float('nan')

        datos_ordenados = sorted(self.datos)
        mitad = n // 2

        return (datos_ordenados[mitad - 1] + datos_ordenados[mitad]) / 2 if n % 2 == 0 else datos_ordenados[mitad]

    def moda(self):
        """Calcula la moda (valor más frecuente)."""
        n = self.contar_datos()
        if n == 0:
            return []

        frecuencias = {}
        for valor in self.datos:
            frecuencias[valor] = frecuencias.get(valor, 0) + 1

        max_frecuencia = max(frecuencias.values())

        # Si todos los valores son únicos, no hay moda
        if max_frecuencia == 1 and n > 1:
            return []

        modas = [valor for valor, freq in frecuencias.items() if freq == max_frecuencia]
        return modas[0] if len(modas) == 1 else modas

    def varianza(self):
        """Calcula la varianza muestral."""
        n = self.contar_datos()
        if n < 2:
            return float('nan')

        media = self.media()
        suma_cuadrados = sum((valor - media) ** 2 for valor in self.datos)
        return suma_cuadrados / (n - 1)

    def desviacion_estandar(self):
        """Calcula la desviación estándar como raíz cuadrada de la varianza."""
        return np.sqrt(self.varianza())
def rango(self):
    """Devuelve el rango estadístico: diferencia entre el valor máximo y mínimo."""
    if not self.datos:
        return float('nan')  # No se puede calcular el rango sin datos

    return max(self.datos) - min(self.datos)

def coeficiente_variacion(self):
    """
    Calcula el Coeficiente de Variación (CV) de Pearson.
    El CV mide la dispersión relativa como porcentaje de la media.
    """
    media = self.media()
    desviacion = self.desviacion_estandar()

    # Validación: si media o desviación estándar no son válidas
    if np.isnan(media) or np.isnan(desviacion):
        return float('nan')

    # Casos especiales: media igual a cero
    if media == 0:
        # Si todos los datos son cero, no hay variabilidad
        if desviacion == 0:
            return 0.0
        # Si hay variabilidad pero la media es cero, el CV es indefinido (infinito)
        return float('inf')

    # Cálculo estándar del CV como porcentaje
    return (desviacion / media) * 100