# Directory

```
├── notebook
│   ├── cat_image_process.ipynb
│   └── esrgan.ipynb
└── input
    └── cat-dataset
```

# Env

colab or docker

# Data

put cat-dataset into input

https://drive.google.com/file/d/1LO-uE69XSb1SOrzpQZN6WBDfpvyKkMOQ/view?usp=sharing

# Data process

cat_image_process.ipynb

# Train

python esrgan.ipynb

# Result

![image](https://user-images.githubusercontent.com/34574033/82691642-2ef5a000-9c99-11ea-9b56-504e95c54b43.png)

![000024050](https://user-images.githubusercontent.com/34574033/82691665-3cab2580-9c99-11ea-838e-9034ebd06994.png)

![0000656](https://user-images.githubusercontent.com/34574033/82691876-9dd2f900-9c99-11ea-805f-66417c1fbe03.jpg)


=========== 以下 memo ===========

```
python esrgan/train.py --batch_size 1 --sample_interval 1 --warmup_batches 1 --hr_height 160 --hr_width 160
```

## check
```
python esrgan/train.py \
--batch_size 1 \
--sample_interval 1 \
--warmup_batches 1
```

# Demo

## mlflow
export PATH=$PATH:/home/ubuntu/.local/bin/ export LC_ALL=C.UTF-8 export LANG=C.UTF-8 mlflow ui
mlflow ui
