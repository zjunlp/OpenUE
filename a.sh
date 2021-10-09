conda create -n openue python=3.8
conda activate openue
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia # 视自己Nvidia驱动环境选择对应的cudatoolkit版本
python setup.py install
