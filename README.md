# Env


# Train


## check
python esrgan/train.py \
--batch_size 1 \
--sample_interval 1 \
--warmup_batches 1

# Demo

## mlflow
export PATH=$PATH:/home/ubuntu/.local/bin/ export LC_ALL=C.UTF-8 export LANG=C.UTF-8 mlflow ui
mlflow ui
