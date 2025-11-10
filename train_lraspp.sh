# lraspp
(python train.py --dataloader='OPA_Dist' \
--img_dir='dataset/OPA/new_OPA/background' \
--label_dir='dataset/OPA/new_OPA/new_train_label.csv' \
--test_label_dir='dataset/OPA/new_OPA/new_test_label.csv' \
--model='ynet' \
--loss='dice' \
--batch_size="8" \
--obj_dir='dataset/OPA/new_OPA/foreground' \
--mask_dir='dataset/OPA/new_OPA/max' \
--lr='0.0025' --optim='adam' --epoch='25')
