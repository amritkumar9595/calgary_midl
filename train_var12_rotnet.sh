BATCH_SIZE=1
NUM_EPOCHS=70
DEVICE='cuda:0'
SAMPLE_RATE=47
CASCADE=12


                                            ## 32 channel , R = 5x data ##
<<ACC_FACTOR_5x

RESUME='False'
LR=0.000001
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/pretext/acc_'${ACC}
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/pretext/acc_'${ACC}'/model.pt'

python train_var12_pretext_rotnet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT}  --lr ${LR} --cascade ${CASCADE}
ACC_FACTOR_5x

                                                ## Rotnet + slicemiss in alternative training
<<ACC_FACTOR_5x
RESUME='False'
LR2=0.000001
LR1=0.001
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/slicemiss_rotnet/pretext/acc_'${ACC}
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/slicemiss_rotnet/pretext/acc_'${ACC}'/model.pt'

python train_var12_pretext_slicemiss_rotnet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT}  --lr1 ${LR1} --lr2 ${LR2} --cascade ${CASCADE}
ACC_FACTOR_5x

                                                ## multi-pretext learning 

#<<ACC_FACTOR_5x
RESUME='False'
LR=0.001
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/multi/pretext/acc_'${ACC}
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/multi/pretext/acc_'${ACC}'/model.pt'

python train_var12_pretext_multitask.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT}  --lr ${LR}  --cascade ${CASCADE}
#ACC_FACTOR_5x








                                            ## 12 channel data  FINETUNING  ##

<<ACC_FACTOR_5x
RESUME='Fasle'
LAYER_INIT=5  # 1 gives better results
LR=0.001
LOSS='SSIM'  # MSE,SSIM
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/pretext/acc_5x/best_model_rot.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/finetune/acc_'${ACC}'/'${LAYER_INIT}'_layer/'${LR}'_lr/'${SAMPLE_RATE}'_volume'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/finetune/acc_'${ACC}'/'${LAYER_INIT}'_layer/'${LR}'_lr/'${SAMPLE_RATE}'_volume/model.pt'

python train_var12_finetune_rotnet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}  --pretext-model ${PRETEXT_MODEL} --lr ${LR} --layer-init ${LAYER_INIT} --cascade ${CASCADE}
ACC_FACTOR_5x


                                            ## 12 channel data  FINETUNING ON ROTNET + AUTOENCODER ##

<<ACC_FACTOR_5x
PRETEXT2='autoencoder'
RESUME='True'
LAYER_INIT=0  # 1 gives better results
LR=0.001
LOSS='SSIM'  # MSE,SSIM
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=5
ACC='5x'
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/pretext/'${PRETEXT2}'/acc_5x/12_cascade/0.001_lr/best_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/finetune/'${PRETEXT2}'/acc_'${ACC}'/'${LAYER_INIT}'_layer/'${LR}'_lr/'${SAMPLE_RATE}'_volume'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet_unet/12-channels/rotnet/finetune/'${PRETEXT2}'/acc_'${ACC}'/'${LAYER_INIT}'_layer/'${LR}'_lr/'${SAMPLE_RATE}'_volume/model.pt'

python train_var12_finetune_rotnet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS}  --pretext-model ${PRETEXT_MODEL} --lr ${LR} --layer-init ${LAYER_INIT} --cascade ${CASCADE}
ACC_FACTOR_5x





