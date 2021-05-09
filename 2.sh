python train.py --arch cnn --lr 1e-6 --optim adam --pre_trained no --data_aug no --gpu 0
python train.py --arch cnn --lr 1e-6 --optim adam --pre_trained yes --data_aug no --gpu 0
python train.py --arch cnn --lr 1e-5 --optim sgd --pre_trained no --data_aug yes --gpu 0
python train.py --arch cnn --lr 1e-5 --optim sgd --pre_trained yes --data_aug yes --gpu 0
python train.py --arch cnn --lr 1e-5 --optim adam --pre_trained no --data_aug yes --gpu 0
python train.py --arch cnn --lr 1e-5 --optim adam --pre_trained yes --data_aug yes --gpu 0
