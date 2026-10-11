import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class RecommendationSystem:
    """
    Sistema de recomendación simple usando KNeighborsClassifier.
    """
    
    def __init__(self, n_neighbors=5):
        """
        Inicializa el sistema de recomendación.
        
        Parameters:
        -----------
        n_neighbors : int
            Número de vecinos cercanos a considerar (default: 5)
        """
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        self.is_trained = False
        
    def load_data(self, filepath):
        """
        Carga los datos desde un archivo CSV.
        
        Parameters:
        -----------
        filepath : str
            Ruta al archivo products.csv
            
        Returns:
        --------
        pd.DataFrame
            DataFrame cargado
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Datos cargados exitosamente desde {filepath}")
            print(f"  Shape: {df.shape}")
            print(f"  Columnas: {df.columns.tolist()}\n")
            return df
        except FileNotFoundError:
            print(f"✗ Error: Archivo {filepath} no encontrado")
            return None
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepara los datos dividiendo en entrenamiento y prueba.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con columnas: feature1, feature2, feature3, label
        test_size : float
            Proporción de datos para prueba (default: 0.2)
        random_state : int
            Seed para reproducibilidad
        """
        # Verificar que las columnas necesarias existan
        required_columns = ['feature1', 'feature2', 'feature3', 'label']
        if not all(col in df.columns for col in required_columns):
            print(f"✗ Error: El DataFrame debe contener las columnas: {required_columns}")
            return False
        
        # Separar features y labels
        self.feature_columns = ['feature1', 'feature2', 'feature3']
        X = df[self.feature_columns].values
        y = df['label'].values
        
        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ Datos preparados:")
        print(f"  Entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"  Prueba: {self.X_test.shape[0]} muestras")
        print(f"  Features: {self.feature_columns}\n")
        
        return True
    
    def train(self):
        """
        Entrena el modelo KNeighborsClassifier.
        """
        if self.X_train is None:
            print("✗ Error: Primero debe preparar los datos con prepare_data()")
            return False
        
        try:
            self.model.fit(self.X_train, self.y_train)
            self.is_trained = True
            print(f"✓ Modelo entrenado exitosamente")
            print(f"  Algoritmo: KNeighborsClassifier")
            print(f"  n_neighbors: {self.n_neighbors}\n")
            return True
        except Exception as e:
            print(f"✗ Error durante el entrenamiento: {e}")
            return False
    
    def evaluate(self):
        """
        Evalúa el modelo en el conjunto de prueba.
        
        Returns:
        --------
        dict
            Diccionario con resultados de evaluación
        """
        if not self.is_trained:
            print("✗ Error: El modelo no ha sido entrenado aún")
            return None
        
        # Predicciones
        y_pred = self.model.predict(self.X_test)
        
        # Calcular accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"✓ Evaluación del modelo:")
        print(f"  Accuracy en prueba: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        
        # Reporte de clasificación
        print("Reporte de clasificación:")
        print(classification_report(self.y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': self.y_test
        }
    
    def recommend(self, features, n_recommendations=5):
        """
        Genera recomendaciones basadas en features similares.
        
        Parameters:
        -----------
        features : list or array
            Vector de características [feature1, feature2, feature3]
        n_recommendations : int
            Número de recomendaciones a retornar
            
        Returns:
        --------
        dict
            Diccionario con la predicción y distancias
        """
        if not self.is_trained:
            print("✗ Error: El modelo no ha sido entrenado aún")
            return None
        
        # Convertir a array numpy
        features_array = np.array(features).reshape(1, -1)
        
        # Realizar predicción
        prediction = self.model.predict(features_array)[0]
        
        # Obtener distancias y índices de los vecinos más cercanos
        distances, indices = self.model.kneighbors(features_array)
        
        recommendation = {
            'input_features': features,
            'predicted_label': prediction,
            'confidence': f"{(1 - distances.mean()) * 100:.2f}%",
            'nearest_neighbors': {
                'distances': distances[0].tolist(),
                'neighbor_labels': self.y_train[indices[0]].tolist()
            }
        }
        
        return recommendation
    
    def print_recommendation(self, recommendation):
        """
        Imprime una recomendación de forma legible.
        
        Parameters:
        -----------
        recommendation : dict
            Diccionario con la recomendación
        """
        if recommendation is None:
            return
        
        print(f"📊 Recomendación:")
        print(f"  Features de entrada: {recommendation['input_features']}")
        print(f"  Categoría predicha: {recommendation['predicted_label']}")
        print(f"  Confianza: {recommendation['confidence']}")
        print(f"  Vecinos más cercanos:")
        for i, (dist, label) in enumerate(zip(
            recommendation['nearest_neighbors']['distances'],
            recommendation['nearest_neighbors']['neighbor_labels']
        ), 1):
            print(f"    {i}. Distancia: {dist:.4f}, Categoría: {label}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SISTEMA DE RECOMENDACIÓN - KNeighborsClassifier")
    print("=" * 70 + "\n")
    
    # 1. Crear instancia del sistema
    rec_system = RecommendationSystem(n_neighbors=5)
    
    # 2. Cargar datos
    df = rec_system.load_data('products.csv')
    
    if df is not None:
        # 3. Preparar datos (dividir en entrenamiento y prueba)
        rec_system.prepare_data(df, test_size=0.2)
        
        # 4. Entrenar el modelo
        rec_system.train()
        
        # 5. Evaluar el modelo
        results = rec_system.evaluate()
        
        # 6. Hacer recomendaciones
        print("=" * 70)
        print("EJEMPLOS DE RECOMENDACIONES")
        print("=" * 70 + "\n")
        
        # Ejemplo 1: Usar características del primer sample de prueba
        sample_features = rec_system.X_test[0].tolist()
        rec1 = rec_system.recommend(sample_features)
        rec_system.print_recommendation(rec1)
        print()
        
        # Ejemplo 2: Usar características del segundo sample de prueba
        sample_features = rec_system.X_test[1].tolist()
        rec2 = rec_system.recommend(sample_features)
        rec_system.print_recommendation(rec2)
        print()
        
        # Ejemplo 3: Usar features personalizadas
        custom_features = [5.0, 3.5, 2.0]
        rec3 = rec_system.recommend(custom_features)
        rec_system.print_recommendation(rec3)
