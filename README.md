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
