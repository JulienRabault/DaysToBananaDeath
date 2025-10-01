#!/usr/bin/env python3
"""
Script de test simple pour l'API Banana Classifier
"""

import requests
import time

def test_api():
    """Test simple de l'API"""
    base_url = "http://localhost:8000"

    print("🍌 Test de l'API Banana Classifier")
    print("=" * 40)

    # Test 1: Page d'accueil
    try:
        response = requests.get(f"{base_url}/")
        print("✅ Page d'accueil:", response.json()["message"])
    except Exception as e:
        print("❌ Erreur page d'accueil:", e)
        return

    # Test 2: Health check (avec attente pour le chargement du modèle)
    print("\n⏳ Vérification du chargement du modèle...")
    for i in range(10):  # Attendre jusqu'à 30 secondes
        try:
            response = requests.get(f"{base_url}/health")
            health = response.json()

            if health["model_loaded"]:
                print("✅ Modèle chargé avec succès!")
                break
            else:
                print(f"⏳ Attente du modèle... ({i+1}/10)")
                time.sleep(3)
        except Exception as e:
            print(f"❌ Erreur health check: {e}")
            time.sleep(3)
    else:
        print("❌ Le modèle n'a pas pu être chargé dans les temps")
        return

    print("\n🎉 API prête à recevoir des images de bananes!")
    print("\nPour tester avec une image:")
    print("curl -X POST http://localhost:8000/predict -F 'file=@votre_image.jpg'")
    print("\nOu utilisez la documentation interactive: http://localhost:8000/docs")

if __name__ == "__main__":
    test_api()
