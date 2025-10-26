from flask import Blueprint, request, jsonify
from .query_data import process_query  # Importamos la función process_query de query_data

# Crea un Blueprint para las rutas
api = Blueprint("api", __name__)

@api.route("/query", methods=["POST"])
def query():
    try:
        # Obtén el prompt del cuerpo de la solicitud
        data = request.get_json()
        prompt = data.get("prompt")  # 'prompt' es el campo que se espera en el JSON
        if not prompt:
            return jsonify({"error": "El campo 'prompt' es requerido"}), 400

        # Llama a la función process_query, pasando el 'prompt'
        response = process_query(prompt)

        # Devuelve el resultado
        return jsonify(response), 200
    except Exception as e:
        # Manejo de errores
        return jsonify({"error": str(e)}), 500
