
import importlib
from color_generator.color_predictor import ColorPredictor

datasets_module = importlib.import_module("datasets", package="color_generator")
dataset_class_ = getattr(datasets_module, 'ColorsDataset')
dataloader_class_ = getattr(datasets_module, "DataLoaders")

dataset_args = {}
dataset = dataset_class_(**dataset_args)
dataloaders = dataloader_class_(dataset)

networks_module = importlib.import_module("color_generator.networks")
network_fn_ = getattr(networks_module, 'Distilbert')
network_args = {}
network = network_fn_(**network_args)


models_module = importlib.import_module("color_generator.models")
model_class_ = getattr(models_module, 'ColorModel')

model = model_class_(
    dataloaders=dataloaders, network_fn=network, device='cpu'
)
model = ColorPredictor(model)
pred = model.predict_color('light skin', plot=False)
print(pred)
