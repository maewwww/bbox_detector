# lraspp
(python train.py --dataloader='OPA_Dist' \
--img_dir='dataset/OPA/new_OPA/background' \
--label_dir='dataset/OPA/new_OPA/new_train_label.csv' \
--test_label_dir='dataset/OPA/new_OPA/new_test_label.csv' \
--model='lraspp' \
--loss='kldiv' \
--batch_size="32" \
--obj_dir='dataset/OPA/new_OPA/foreground' \
--mask_dir='dataset/OPA/new_OPA/weighted' \
--lr='0.00125' --optim='adam' --epoch='100')
