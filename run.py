import os
import webbrowser
import threading
from app import app

def ouvrir_navigateur():
    """Ouvre le navigateur après le démarrage du serveur"""
    import time
    time.sleep(2)  # Attendre que le serveur démarre
    webbrowser.open('http://localhost:5000')

if __name__ == '_main_':
    print("🚀 Démarrage de l'application de validation d'emails...")
    print("📧 Patientez pendant le chargement...")
    
    # Ouvrir le navigateur automatiquement
    threading.Thread(target=ouvrir_navigateur).start()
    
    # Démarrer l'application Flask
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)