from color_generator.color_predictor import ColorPredictor

from tensorflow.keras.models import load_model

reconstructed_model = load_model("distilbert_test_model")

model = ColorPredictor()
prediction = model.predict("blue")

prediction2 = reconstructed_model.predict("yellow")

prediction3 = reconstructed_model.predict("blue")

print(prediction, prediction2, prediction3)