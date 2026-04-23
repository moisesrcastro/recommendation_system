# src/recommender/model.py

import os
import tensorflow as tf


def carregar_modelo():
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )

    caminho_modelo = os.path.join(
        base_dir,
        "model",
        "neural_network.keras"
    )

    modelo = tf.keras.models.load_model(
        caminho_modelo
    )

    return modelo