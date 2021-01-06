BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
RESUME='False'
LR=0.000001

SAMPLE_RATE=0.1

                                            ## 32 channel , R = 5x data ##
#<<ACC_FACTOR_5x
LOSS='SSIM'
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/Test/test_32_channel/Test-R=5" 
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/chal_exp/varnet/acc_'${ACC}'/12-channels/'${LOSS}'_loss'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/chal_exp/varnet/acc_'${ACC}'/12-channels/'${LOSS}'_loss/vs_model.pt'

python train_var32_rotnet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT} --loss ${LOSS} --lr ${LR}
#ACC_FACTOR_5x





<<ACC_FACTOR_10x
DATA_PATH="/media/student1/RemovableVolume/calgary/"
ACC_FACTOR=10
ACC='10x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/chal_exp/dropout/12_channel/acc_'${ACC}'/'
CHECKPOINT="/media/student1/NewVolume/MR_Reconstruction/experiments/chal_exp/dropout/12_channel/acc_5x/vs_model.pt"

python train_vs.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT}
ACC_FACTOR_10x

                                            ##32 channel data ##
<<ACC_FACTOR_5x
DROPOUT=0.0
DATA_PATH="/media/student1/RemovableVolume/calgary/32_channels_218_180" 
DATA_PATH2="/media/student1/RemovableVolume/calgary/12_channels_218_180" 
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/chal_exp/varnet/acc_'${ACC}'/32-channels'
CHECKPOINT="/media/student1/NewVolume/MR_Reconstruction/experiments/chal_exp/varnet/acc_5x/32-channels/vs_model.pt"
python train_vs_sens.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --data-path2 ${DATA_PATH2} --resume ${RESUME} --checkpoint ${CHECKPOINT} --dropout ${DROPOUT}
ACC_FACTOR_5x

<<ACC_FACTOR_10x
DATA_PATH="/media/student1/RemovableVolume/calgary/calgary_32_additional_data/"
ACC_FACTOR=10
ACC='10x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/chal_exp/dropout/32_channel/acc_'${ACC}'/'
CHECKPOINT="/media/student1/NewVolume/MR_Reconstruction/experiments/chal_exp/dropout/32_channel/acc_10x/vs_model.pt"

python train_vs.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT}
ACC_FACTOR_10x




