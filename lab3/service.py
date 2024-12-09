from torchvision import transforms
from PIL import Image
import bentoml
from bentoml.io import Image as BentoImage, JSON
import numpy as np
import asyncio
import nest_asyncio
nest_asyncio.apply()

# Инициализация модели
cifar100_runner = bentoml.onnx.get("model:latest").to_runner()
svc = bentoml.Service("model", runners=[cifar100_runner])

# API для предсказаний
@svc.api(input=BentoImage(), output=JSON())
async def predict(img: Image.Image):
    classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    # Преобразование изображения
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация как в обучении
    ])
    img_tensor = transform(img).unsqueeze(0).numpy()  # Добавляем batch dimension

    loop = asyncio.get_event_loop()
    predictions = loop.run_until_complete(cifar100_runner.run.async_run(img_tensor))
    
    print(f"Предсказания модели: {predictions}")
    predicted_class = int(np.argmax(predictions[0]))
    print({"class_id": predicted_class, "class_name": classes[predicted_class]})
    return {"class_id": predicted_class, "class_name": classes[predicted_class]}