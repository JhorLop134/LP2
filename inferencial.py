import pandas as pd
from scipy import stats # Necesario para valores críticos (t y Z)
# Asume que tu clase base extendida está en un archivo importable
from base import EstadisticaBase 

class InferenciaEstadistica(EstadisticaBase):
    """
    Clase hija para realizar inferencia estadística (Intervalos de Confianza) 
    utilizando métodos de la clase EstadisticaBase.
    """
    
    def __init__(self, datos):
        """Inicializa la clase base y asegura que haya suficientes datos."""
        super().__init__(datos)
        self._n = self.obtener_n_observaciones()
        
        if self._n < 2:
             raise ValueError("Se requieren al menos 2 observaciones para la inferencia.")

    # --- Métodos de Inferencia para la Media (Cuantitativa) ---
    
    def intervalo_confianza_media(self, nivel_confianza=0.95):
        """
        Calcula el Intervalo de Confianza (IC) para la media poblacional.
        Utiliza la distribución t de Student (basado en la desviación estándar muestral).
        
        :param nivel_confianza: Nivel de confianza (ej: 0.95 para 95%).
        :return: Tupla (Límite Inferior, Límite Superior).
        """
        datos_cuantitativos = self.obtener_datos().astype(float) # Aseguramos tipo numérico
        
        try:
            media = self.media()
            ds = self.desviacion_estandar() 
        except TypeError:
            raise TypeError("Los datos deben ser numéricos para calcular el IC de la media.")
            
        grados_libertad = self._n - 1
        
        # Valor crítico t de Student: (1 - (1 - 0.95)/2) = 0.975 para 95%
        t_critico = stats.t.ppf(1 - (1 - nivel_confianza) / 2, grados_libertad)
        
        # Error Estándar de la Media
        error_estandar = ds / (self._n ** 0.5)
        
        # Margen de Error
        margen_error = t_critico * error_estandar
        
        limite_inferior = media - margen_error
        limite_superior = media + margen_error
        
        return (limite_inferior, limite_superior)

    # --- Métodos de Inferencia para la Proporción (Cualitativa) ---

    def intervalo_confianza_proporcion(self, valor_exito, nivel_confianza=0.95):
        """
        Calcula el Intervalo de Confianza para la proporción poblacional (IC).
        Asume una distribución Normal (aproximación Z).
        
        :param valor_exito: El valor categórico que se considera 'éxito' (ej: 'Femenino').
        :param nivel_confianza: Nivel de confianza (ej: 0.95 para 95%).
        :return: Tupla (Límite Inferior, Límite Superior).
        """
        datos_categoricos = self.obtener_datos()
        
        # 1. Obtener la Proporción Muestral (p_hat)
        conteo_exito = (datos_categoricos == valor_exito).sum()
        p_hat = conteo_exito / self._n
        q_hat = 1 - p_hat
        
        # 2. Valor crítico Z (distribución normal estándar)
        z_critico = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)

        # 3. Error Estándar de la Proporción
        # np.sqrt(p_hat * q_hat / self._n)
        error_estandar = (p_hat * q_hat / self._n) ** 0.5
        
        # 4. Margen de Error
        margen_error = z_critico * error_estandar
        
        limite_inferior = p_hat - margen_error
        limite_superior = p_hat + margen_error
        
        return (limite_inferior, limite_superior)

    # --- Polimorfismo / Implementación de resumen() ---
    
    def resumen(self):
        """
        Proporciona un resumen de las inferencias clave disponibles.
        En este caso, devuelve el IC de la media.
        """
        res = {
            "Conteo": self._n,
        }
        
        # Intenta calcular el IC de la media solo si los datos son numéricos
        try:
             ic_media = self.intervalo_confianza_media()
             res["Media Muestral"] = self.media()
             res[f"IC Media ({int(0.95*100)}%)"] = ic_media
        except TypeError:
             res["IC Media"] = "No aplicable (Datos no numéricos)"
        except Exception as e:
             res["IC Media"] = f"Error al calcular: {e}"

        return res

    def __str__(self):
        """Representación de string de la clase."""
        res = self.resumen()
        
        output = f"Inferencia Estadística (n={self._n})\n"
        output += "-"*30 + "\n"
        
        if "Media Muestral" in res:
             media = res.pop("Media Muestral")
             ic = res.pop(f"IC Media ({int(0.95*100)}%)")
             output += f"Media Muestral: {media:.4f}\n"
             output += f"IC de la Media (95%): ({ic[0]:.4f}, {ic[1]:.4f})\n"
        else:
             output += res["IC Media"] + "\n"
             
        output += "-"*30 + "\n"
        output += "Utilice .intervalo_confianza_proporcion(valor) para IC de proporción."
        
        return output