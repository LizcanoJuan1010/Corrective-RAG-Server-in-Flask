from flask import Flask
from app.routes import api  # Importa correctamente el Blueprint desde app.routes

def create_app():
    app = Flask(__name__)
    app.register_blueprint(api)  # Registra el Blueprint definido en routes.py
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True)
