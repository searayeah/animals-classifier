# Animals-classifier

- Задача классификация картинок на кошек и собак.
- Использую resnet18 с предобученными весами. Обучаю только последний слой.
- Обучение `python animals_classifier/train.py`
- Инференс `python animals_classifier/infer.py`
- Инференс предсказывает класс на тестовых данных и выводит предсказания в `pred.csv` в корень проекта.
- Датасеты подгружаются с помощью `python animals_classifier/dataset.py`
- Модель подгружается с помощью `python animals_classifier/model.py`
- Логирую три параметра в MLFlow во время обучения.
  ![image](https://github.com/searayeah/animals-classifier/assets/57370975/a034c24c-6846-463e-92df-997ae9791658)
