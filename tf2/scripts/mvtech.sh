categories=[Bottle,Pill,Tile,Transistor,Zipper,Hazelnut,metal_nut,Toothbrush,Wood]
run_id=001
batch_size=256
image_size=224

python3 run.py --dataset=mvtech --data_dir="/mnt/runs/data" --eval_split=test \
--model_dir="/mnt/runs/logs" --mode=train_then_eval --train_mode=pretrain --image_size=$image_size \
--image_size=$image_size --train_batch_size=$batch_size --lineareval_while_pretraining=False --test_perc=0.15 \
--eval_batch_size=$batch_size --anomaly_perc=0.1 --run_id=$run_id \
--train_epochs=1000 # --categories=$categories

python3 run.py --dataset=mvtech --data_dir="/mnt/runs/data" --eval_split=test \
--model_dir="/mnt/runs/logs" --mode=train_then_eval --train_mode=finetune --image_size=$image_size \
--image_size=$image_size --train_batch_size=$batch_size --test_perc=0.5 \
--eval_batch_size=$batch_size --anomaly_perc=0.5 --run_id=$run_id \
--train_epochs=1000 # --categories=$categories