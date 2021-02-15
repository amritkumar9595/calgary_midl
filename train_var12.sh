BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'

SAMPLE_RATE=2
                                            ## 12 channel data  TRAINING FROM SCRATCH ##

#<<ACC_FACTOR_5x
LR=0.001
CASCADE=12
RESUME='False'
LOSS='SSIM'
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/scratch/acc_5x/'${SAMPLE_RATE}'_volume/'${CASCADE}'_cascade/'${LR}'_lr'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/scratch/acc_5x/'${SAMPLE_RATE}'_volume/'${CASCADE}'_cascade/'${LR}'_lr/model.pt'

python train_var12_scratch.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS} --cascade ${CASCADE}
#ACC_FACTOR_5x



                                            ## 12 channel data  PRETEXT TRAINING  ##

<<ACC_FACTOR_5x
RESUME='False'
LOSS='MSE'  # SSIM
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/pretext/acc_5x/nodc'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/pretext/acc_5x/nodc/model.pt'

python train_var12_pretext.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}
ACC_FACTOR_5x


                                            ## 12 channel data  FINETUNING  ##

<<ACC_FACTOR_5x
RESUME='False'
LAYER_INIT=0  # 1 gives better results
LR=0.001
LOSS='SSIM'  # MSE,SSIM
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/pretext/acc_5x/nodc/best_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/finetune/acc_'${ACC}'/'${LAYER_INIT}'_layer/'${LR}'/nodc/'${SAMPLE_RATE}'_volume'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/12-channels/finetune/acc_'${ACC}'/'${LAYER_INIT}'_layer/'${LR}'/nodc/'${SAMPLE_RATE}'_volume/model.pt'

python train_var12_finetune.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}  --pretext-model ${PRETEXT_MODEL} --lr ${LR} --layer-init ${LAYER_INIT}
ACC_FACTOR_5x




