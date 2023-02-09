module load python
module load scipy-stack
module load mpi4py/3.1.3
virtualenv --no-download .env
source .env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index matplotlib Pillow scikit-learn scipy opencv_python imageio tensorboard 
pip install --no-index tqdm seaborn msgpack PyYAML ConfigArgParse urllib3
pip install scikit-spatial 
pip install -e .
