# fusion 2
(python train.py --dataloader='OPA_3' \
--img_dir='dataset/OPA/new_OPA/background' \
--label_dir='dataset/OPA/new_OPA/new_alt_train_label.csv' \
--test_label_dir='dataset/OPA/new_OPA/new_alt_test_label.csv' \
--model='fusion_2' \
--loss='var_mse_min' \
--batch_size="12" \
--obj_dir='dataset/OPA/new_OPA/foreground' \
--mask_dir='dataset/OPA/new_OPA/max' \
--lr='0.001' --optim='adam' --epoch='50')
