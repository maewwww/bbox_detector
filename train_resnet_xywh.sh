# OPA
(python train.py --dataloader='OPA' \
--img_dir='dataset/OPA/new_OPA/background' \
--label_dir='dataset/OPA/new_OPA/new_train_label.csv' \
--test_label_dir='dataset/OPA/new_OPA/new_test_label.csv' \
--model='double_resnet50' \
--loss='var_mse_min' \
--batch_size="16" \
--obj_dir='dataset/OPA/new_OPA/foreground' \
--lr='0.001' --optim='adam' --epoch='60')
