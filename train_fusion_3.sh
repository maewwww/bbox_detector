# fusion 3
(python train.py --dataloader='OPA_2' \
--img_dir='dataset/OPA/new_OPA/background' \
--label_dir='dataset/OPA/new_OPA/new_alt_train_label.csv' \
--test_label_dir='dataset/OPA/new_OPA/new_alt_test_label.csv' \
--model='fusion_3' \
--loss='matching' \
--batch_size="16" \
--obj_dir='dataset/OPA/new_OPA/foreground' \
--mask_dir='dataset/OPA/new_OPA/max' \
--lr='0.00125' --optim='adam' --epoch='50')
