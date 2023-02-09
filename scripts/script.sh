MODEL_FLAGS="--dataset rplan --batch_size 512 --set_name train --target_set 8"
TRAIN_FLAGS="--lr 1e-3 --save_interval 5000 --weight_decay 0.05 --log_interval 500"
SAMPLE_FLAGS="--batch_size 64 --num_samples 64"

CUDA_VISIBLE_DEVICES='0' python image_train.py $MODEL_FLAGS $TRAIN_FLAGS
#CUDA_VISIBLE_DEVICES='1' python image_sample.py $MODEL_FLAGS --model_path ckpts/exp/model250000.pt $SAMPLE_FLAGS
