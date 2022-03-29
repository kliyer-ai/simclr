categories=[Bottle,Pill,Tile,Transistor,Zipper,Hazelnut,metal_nut,Toothbrush,Wood]

python3 run.py --dataset=mvtech --data_dir="/mnt/dev/data" --eval_split=test \
--model_dir="/mnt/dev/runs/logs" --mode=train_then_eval --train_mode=pretrain --image_size=224 \
--image_size=224 --train_batch_size=256 --lineareval_while_pretraining=True --test_perc=0.5 \
--eval_batch_size=256 --load_existing_split=False --anomaly_perc=0.15 --run_id=001 \
--train_epochs=200 --eval_mahal=True # --categories=$categories



# --train_mode= pretrain or finetune
# it's (obviously) necessary to pretrain first because the models loads a checkpoint.

# I guess for our project we would only pretrain the model

# we then need to think about how to adapt the task for anomaly detection
# instead of a projection head, I guess we could use Conv2DTranspose to restore the original image
# and then compute a reconstruction loss over input and output


# steps
# import dataset to be class of TFDS builder
# pretrain with that model
# see what happens
# add upscaling head and compute reconstruction loss (finetune)