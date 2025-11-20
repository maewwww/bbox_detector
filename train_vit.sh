# vit
(python train.py --dataloader='OPA' \
--img_dir='dataset/OPA/new_OPA/background' \
--label_dir='dataset/OPA/new_OPA/new_alt_train_label.csv' \
--test_label_dir='dataset/OPA/new_OPA/new_alt_test_label.csv' \
--model='b_vit' \
--loss='var_mse_min' \
--batch_size="32" \
--obj_dir='dataset/OPA/new_OPA/foreground' \
--mask_dir='dataset/OPA/new_OPA/max' \
--lr='0.005' --optim='adam' --epoch='50')
