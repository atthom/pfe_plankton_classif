python3 -m virtualenv virtual_env_py3_naif
cd virtual_env_py3_naif
source bin/activate
cp ../gen_features_* ../start_imgnet.sh ../sub_script.sh .
module load cuda/8.0
module load cudnn/5.1-cuda-8.0

#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl

pip3 install --upgrade pip
pip3 install tensorflow-gpu
pip3 install keras h5py pillow
#pip3 install $TF_BINARY_URL

python3 gen_features_naive.py
