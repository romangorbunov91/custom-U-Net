# Создание и оптимизация ResNet18
Поэтапная разработка кастомной ResNet18 модели-классификатора с анализом влияния различных архитектурных решений на производительность.

## Часть 1: Подготовка данных
Создан датакласс [TinyImageNetDataset.py](src\datasets\TinyImageNetDataset.py), наследующий от `torch.utils.data.Dataset` следующие методы:
- `__init__`: инициализация путей к данным и аннотациям, загрузка тренировочного и валидационного датасетов по выбранным классам;
- `__len__`: возврат количества примеров в датасете;
- `__getitem__`: загрузка и возврат одного примера (изображение + метка).

## Часть 2: Базовая архитектура ResNet18

В [model_structure.py](src\models\model_structure.py) реализован `class customResNet18` с возможностью инициализации архитектуры модели под следующие входные параметры:
- `num_classes` - количество классов на выходе; например, `num_classes=10`;
- `layers_config` - слои модели в формате списка; например, `[2, 2, 2, 2]` - `"layers_num": 4`, `"block_size": 2`;
- `activation` - функция активации (`ReLU`, `LeakyReLU`, `ELU`, или `GELU`);
- `in_channels` - количество входных каналов; например, для RGB-картинок `in_channels=3`;
- `layer0_channels` - количество каналов на входе первого базового слоя.

### 2.4. Скрипт обучения

#### Конфигурирование проекта
Гиперпараметры задаются в файле [config.json](src\hyperparameters\config.json), включая:
- архитектура модели: `layers_num`, `block_size`, `activation`;
- выбранные классы датасета: `selected_classes`;
- параметры обучения: `epochs`, `batch_size`, `solver`;
- политика обучения: `save_policy` - "all", "best" (политика "early_stop" выбирается установкой параметра "early_stop_number" > 0).

#### Обучение
Обучение реализовано в [train.py](src\train.py) в виде класса `ResNet18Trainer` со следующими методами:
- `__init__` - инициализация переменных класса в соответствии с гиперпараметрами из файла конфигурации проекта;
- `init_model` - установка функции ошибки, инициализация/загрузка модели, загрузка датасета;
- `__train` - обучение по батчам;
- `__val` - валидация по батчам;
- `train` - основной цикл обучения/валидации по эпохам;
- `update_metrics` - аккумулирование losses/accuracy посредством [average_meter.py](src\utils\average_meter.py).

Рекомендуется работать с моделью из терминала посредством [main.py](src\main.py).
```
python src\main.py --hypes src\hyperparameters\config.json 
```
или
```
python src\main.py --hypes src\hyperparameters\config.json --resume checkpoints\tiny-imagenet-200\best_mdl_4x2_ReLU_Adam.pth
```
Логи обучения хранятся в [train_logs](train_logs).

### 2.5. Визуализация базовых результатов

Графики построены в [main_notebook.ipynb](main_notebook.ipynb).

<p align="center" width="100%">
  <img src="./readme_img/loss_acc_4x2_ReLU_Adam.png"
  style="background-color: white; padding: 0;
  width="100%" />
</p>


# Выводы
- В сравнении 

## Reference
- [Полный текст задания](https://github.com/physicorym/designing_neural_network_architectures_2025_01/tree/main/seminar_03)

## Приложения


### Работа с проектом
#### 1. Скачайте файлы репозитория
#### 2. Скачайте датасет [tiny-imagenet-200](https://disk.yandex.ru/d/adWo9fVCLuVQ0Q)
#### 3. Создайте окружение в директории `.venv`
```
python -m venv .venv
```
#### 4. Активируйте окружение
```
.venv\Scripts\activate
```
#### 5. Установите библиотеки
```
pip install -r requirements.txt
```