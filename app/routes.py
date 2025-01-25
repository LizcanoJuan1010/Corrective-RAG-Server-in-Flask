from flask import Blueprint, request, jsonify
from .query_data import process_query  # Importamos la función process_query de query_data
import os
# Crea un Blueprint para las rutas
api = Blueprint("api", __name__)
UPLOAD_FOLDER = './app/data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
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

@api.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se encontró el archivo"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "El nombre del archivo está vacío"}), 400

        # Verificar que el archivo sea un PDF
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Solo se permiten archivos PDF"}), 400

        # Guardar el archivo en la carpeta 'data'
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        return jsonify({"message": "Archivo subido exitosamente", "file_path": file_path}), 200
    except Exception as e:
        # Manejo de errores
        return jsonify({"error": str(e)}), 500