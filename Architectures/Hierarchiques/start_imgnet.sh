python3 -m virtualenv virtual_env_py3
cd virtual_env_py3
source bin/activate
cp ../gen_features_* ../start_imgnet.sh .
module load cuda/8.0
module load cudnn/5.1-cuda-8.0

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl
export TF_BINARY_URL=https://pypi.python.org/packages/f0/a6/0e96918b09081a5ef4224507463c7a2592c1dabbf858379bd1c47be5fc42/tensorflow-1.3.0-cp34-cp34m-manylinux1_x86_64.whl#md5=2f20ce765797c16d2c3e34b973f1d591
export TF_BINARY_URL=https://pypi.python.org/packages/4e/03/78fff4909d2dc52cbd5bc8f2cfeac62a960d190cdfd8c4482f8969477758/tensorflow_gpu-1.3.0-cp34-cp34m-manylinux1_x86_64.whl#md5=4699a017e2c13dcaddaaa3c08d0d9a82
export TF_BINARY_URL=https://pypi.python.org/packages/d1/ac/4cbd5884ec518eb1064ebafed86fcbcff02994b971973d9e4d1c3fe72204/tensorflow_gpu-1.2.0-cp34-cp34m-manylinux1_x86_64.whl#md5=69c299e9af480a1f570279d7e8a0b959

pip3 install keras h5py pillow
pip3 install $TF_BINARY_URL
#pip3 install --upgrade tensorflow-gpu
python3 gen_features_from_imagenet.py
