# Классификация 128x128 + U-Net c бэкбоном


## Датасеты

### Tiny ImageNet-200
Создан датакласс [TinyImageNetDataset.py](src/datasets/TinyImageNetDataset.py), наследующий от `torch.utils.data.Dataset` следующие методы:
- `__init__`: инициализация путей к данным и аннотациям, загрузка тренировочного и валидационного датасетов по выбранным классам;
- `__len__`: возврат количества примеров в датасете;
- `__getitem__`: загрузка и возврат одного примера (изображение + метка).

Размер изменен к 128×128 в `transforms`:

```
self.train_transforms = transforms.Compose([
  transforms.Resize(tuple([int(img_size[1] * 1.125)]*2)),
  transforms.RandomResizedCrop(mdl_img_size[1], scale=(0.8, 1.0)),
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.RandomRotation(10),
  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
  transforms.ToTensor(),
  normalize,
])

self.val_transforms = transforms.Compose([
    transforms.Resize(tuple(mdl_img_size[1:])),
    transforms.ToTensor(),
    normalize,
])
```
где `mdl_img_size = [3, 128, 128]`.

### MOON_SEGMENTATION_BINARY



## Часть 1. Классификатор 128×128

### 2.4. Скрипт обучения

В [???.py](src/models/???.py) реализован `class customResNet18` с возможностью инициализации архитектуры модели под следующие входные параметры:
- `num_classes` - количество классов на выходе; например, `num_classes=10`;
- `layers_config` - слои модели в формате списка; например, `[2, 2, 2]` - `"layers_num": 3`, `"block_size": 2`;
- `activation` - функция активации (`ReLU`, `LeakyReLU`, `ELU`, или `GELU`);
- `in_channels` - количество входных каналов; например, для RGB-картинок `in_channels=3`;
- `layer0_channels` - количество каналов на входе первого базового слоя.

#### Конфигурирование проекта
Гиперпараметры задаются в файле [???.json](src/hyperparameters/???.json), включая:
- количество эпох в конце обучения, на которых включается дообучение backbone: `backbone_tune_epoch`;
- архитектура модели: `layers_num`, `block_size`, `activation`;
- выбранные классы датасета: `selected_classes`;
- параметры обучения: `epochs`, `batch_size`, `solver`;
- политика обучения: `save_policy` - "all", "best" (политика "early_stop" выбирается установкой параметра "early_stop_number" > 0).

#### Обучение
Обучение реализовано в [train.py](src/train.py) в виде класса `ResNet18Trainer` со следующими методами:
- `__init__` - инициализация переменных класса в соответствии с гиперпараметрами из файла конфигурации проекта;
- `init_model` - установка функции ошибки, инициализация/загрузка модели, загрузка датасета;
- `__train` - обучение по батчам;
- `__val` - валидация по батчам;
- `train` - основной цикл обучения/валидации по эпохам;
- `update_metrics` - аккумулирование losses/accuracy посредством [average_meter.py](src/utils/average_meter.py).

Рекомендуется работать с моделью из терминала посредством [main.py](src/main.py).
```
python src\main.py --hypes src\hyperparameters\customResNet-config.json.json
```
```
python src\main.py --hypes src\hyperparameters\customUNet-config.json
```
или
```
python src\main.py --hypes src\hyperparameters\customResNet-config.json --resume checkpoints\customResNet\best_mdl_4x2_ReLU_Adam.pth
```
```
python src\main.py --hypes src\hyperparameters\customUNet-config.json --resume checkpoints\customUNet\best_mdl_4x2_ReLU_Adam.pth
```
Логи обучения хранятся в [train_logs](train_logs).

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
#### 2. Скачайте датасеты
- [tiny-imagenet-200](https://disk.yandex.ru/d/adWo9fVCLuVQ0Q)
- [moon-segmentation-binary](https://disk.yandex.ru/d/bJ6-fjDlVZBNfQ)
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