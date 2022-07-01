conda activate homefun
cd homefun/zhf/cd Classifier
nohup python train.py --pretrained >weights/log3.txt 2>&1 &
