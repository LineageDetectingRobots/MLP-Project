!/bin/sh
SBATCH -N 1	  # nodes requested
SBATCH -n 1	  # tasks requested
SBATCH --partition=Teach-Standard
SBATCH --gres=gpu:1
SBATCH --mem=12000  # memory in Mb
SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}datasets/
export DATASET_DIR=${TMP}datasets/

mkdir -p ${TMP}results/
export RESULTS_DIR=${TMP}results/

mkdir -p ${TMP}models/
export MODELS_DIR=${TMP}models/

mkdir -p ${DATASET_DIR}fiw/
rsync -r ../datasets/fiw/ ${DATASET_DIR}fiw

mkdir -p ${MODELS_DIR}vgg_face_torch/
rsync -r ../models/vgg_face_torch/ ${MODELS_DIR}vgg_face_torch

# Activate the relevant virtual environment:
source ~/.bashrc
conda activate mlp-proj
cd ..
# TODO: Update this later
python framework/networks/vgg_face.py --cluster 1

rsync -r ${RESULTS_DIR} results/
# python train_evaluate_emnist_classification_system.py --batch_size 100 --continue_from_epoch -1 --seed 0 \
                                                    #   --image_num_channels 3 --image_height 32 --image_width 32 \
                                                    #   --dim_reduction_type "strided" --num_layers 4 --num_filters 64 \
                                                    #   --num_epochs 100 --experiment_name 'cifar100_test_exp' \
                                                    #   --use_gpu "True" --gpu_id "0" --weight_decay_coefficient 0. \
                                                    #   --dataset_name "cifar100"