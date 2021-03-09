BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'

SAMPLE_RATE=47
CASCADE=12                                            ## 12 channel data  TRAINING FROM SCRATCH ##

<<ACC_FACTOR_5x
LR=0.001
RESUME='False'
LOSS='SSIM'
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/scratch/acc_5x/'${SAMPLE_RATE}'_volume/'${CASCADE}'_cascade/'${LR}'_lr'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/scratch/acc_5x/'${SAMPLE_RATE}'_volume/'${CASCADE}'_cascade/'${LR}'_lr/model.pt'

python train_var12_scratch_unet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS} --cascade ${CASCADE}
ACC_FACTOR_5x



                                            ## 12 channel data  PRETEXT TRAINING  ##

<<ACC_FACTOR_5x
LR=0.001
RESUME='False'
LOSS='MSE'  # SSIM
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/pretext/acc_5x/'${CASCADE}'_cascade/nodc/'${LR}'_lr/'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/pretext/acc_5x/'${CASCADE}'_cascade/nodc/'${LR}'_lr/model.pt'

python train_var12_pretext_unet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS} --cascade ${CASCADE}
ACC_FACTOR_5x

                                            ## 12 channel data AUTOENCODER PRETEXT TRAINING ON TOP OF OTHER PRETEXT TRAINING  ##

#<<ACC_FACTOR_5x
PRETEXT='rotnet'
LR=0.001
RESUME='False'
LOSS='MSE'  # SSIM
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
CHECKPOINT_PRETEXT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/pretext/acc_5x/best_model_rot.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/pretext/autoencoder/acc_5x/'${CASCADE}'_cascade/'${LR}'_lr/'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/pretext/autoencoder/acc_5x/'${CASCADE}'_cascade/'${LR}'_lr/model.pt'

python train_var12_pretext_aunet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS} --cascade ${CASCADE} --checkpoint-pretext ${CHECKPOINT_PRETEXT} --pretext ${PRETEXT}
#ACC_FACTOR_5x






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
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/pretext/acc_5x/12_cascade/dc/0.001_lr/best_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/finetune/acc_'${ACC}'/'${LAYER_INIT}'_layer/'${LR}'_lr/'${SAMPLE_RATE}'_volume'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/finetune/acc_'${ACC}'/'${LAYER_INIT}'_layer/'${LR}'_lr/'${SAMPLE_RATE}'_volume/model.pt'

python train_var12_finetune_rotnet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}  --pretext-model ${PRETEXT_MODEL} --lr ${LR} --layer-init ${LAYER_INIT} --cascade ${CASCADE}
ACC_FACTOR_5x




