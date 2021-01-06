BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
RESUME='False'
SAMPLE_RATE=0.2
                                            ## 12 channel data  TRAINING FROM SCRATCH ##

#<<ACC_FACTOR_5x
LOSS='SSIM'
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/scratch/acc_'${ACC}'/12-channels/'${LOSS}'_loss'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/scratch/acc_'${ACC}'/12-channels/'${LOSS}'_loss/model.pt'

python train_var12_scratch.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}
#ACC_FACTOR_5x



                                            ## 12 channel data  PRETEXT TRAINING  ##

<<ACC_FACTOR_5x
LOSS='SSIM'  # MSE
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/pretext/acc_'${ACC}'/12-channels/'${LOSS}'_loss'
CHECKPOINT=='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/pretext/acc_'${ACC}'/12-channels/'${LOSS}'_loss/model.pt'

python train_var12_pretext.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}
ACC_FACTOR_5x


                                            ## 12 channel data  FINETUNING  ##

<<ACC_FACTOR_5x
LR=0.0001
LOSS='SSIM'  # MSE
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/pretext/acc_'${ACC}'/12-channels/'${LOSS}'_loss/best_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/finetune/acc_'${ACC}'/12-channels/'${LOSS}'_loss'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/chal_exp/finetune/pretext/'${LOSS}'_loss/model.pt'

python train_var12_finetune.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}  --pretext-model ${PRETEXT_MODEL} --lr ${LR}
ACC_FACTOR_5x




