mkdir -p pretrained_model
cd pretrained_model
wget http://47.92.96.190/model/chinese_wwm_ext_L-12_H-768_A-12.tar.gz  chinese_wwm_ext_L-12_H-768_A-12.tar.gz
tar zxvf chinese_wwm_ext_L-12_H-768_A-12.tar.gz -C chinese_wwm_ext_L-12_H-768_A-12
rm -f chinese_wwm_ext_L-12_H-768_A-12.tar.gz
