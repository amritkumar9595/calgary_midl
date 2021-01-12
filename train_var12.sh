BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'

SAMPLE_RATE=0.1
                                            ## 12 channel data  TRAINING FROM SCRATCH ##

<<ACC_FACTOR_5x
RESUME='True'
LOSS='SSIM'
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/scratch/acc_5x'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/scratch/acc_5x/model.pt'

python train_var12_scratch.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}
ACC_FACTOR_5x



                                            ## 12 channel data  PRETEXT TRAINING  ##

#<<ACC_FACTOR_5x
RESUME='False'
LOSS='SSIM'  # MSE
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/pretext/acc_5x'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/pretext/acc_5x/model.pt'

python train_var12_pretext.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}
#ACC_FACTOR_5x


                                            ## 12 channel data  FINETUNING  ##

<<ACC_FACTOR_5x
LAYER_INIT=0
LR=0.0001
LOSS='SSIM'  # MSE
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/pretext/acc_5x/best_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/finetune/acc_'${ACC}'/'${LAYER_INIT}'_layer'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/finetune/acc_'${ACC}'/'${LAYER_INIT}'_layer/model.pt'

python train_var12_finetune.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}  --pretext-model ${PRETEXT_MODEL} --lr ${LR} --layer-init ${LAYER_INIT}
ACC_FACTOR_5x




