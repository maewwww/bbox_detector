# fusion 3
(python train.py --dataloader='BGDataset' \
--img_dir='dataset/OPA/new_OPA/background' \
--label_dir='dataset/OPA/new_OPA/bg_pretrain_train_label.csv' \
--test_label_dir='dataset/OPA/new_OPA/bg_pretrain_test_label.csv' \
--model='ResnetEncoder' \
--loss='matching' \
--batch_size="16" \
--obj_dir='dataset/OPA/new_OPA/foreground' \
--mask_dir='dataset/OPA/new_OPA/max' \
--lr='0.00001' --optim='adam' --epoch='75')
