BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
RESUME='False'
LR=0.000001
SAMPLE_RATE=1.0

                                            ## 32 channel , R = 5x data ##
#<<ACC_FACTOR_5x
DATA_PATH="/media/student1/RemovableVolume/calgary/Test/test_32_channel/Test-R=5" 
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/pretext/rotnet/32-channels/acc_5x'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/pretext/rotnet/32-channels/acc_5x/vs_model.pt'

python train_var32_rotnet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT}  --lr ${LR}
#ACC_FACTOR_5x



                                            ## 32 channel , R = 10x data ##
<<ACC_FACTOR_5x
DATA_PATH="/media/student1/RemovableVolume/calgary/Test/test_32_channel/Test-R=10" 
ACC_FACTOR=10
ACC='10x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/acc_'${ACC}'/32-channels'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/midl/varnet/acc_'${ACC}'/32-channels/vs_model.pt'

python train_var32_rotnet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT}  --lr ${LR}
ACC_FACTOR_5x





