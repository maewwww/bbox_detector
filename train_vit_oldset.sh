# OPA
(python train.py --dataloader='two_channel' \
--img_dir='dataset/final_voc/img/' \
--label_dir='dataset/final_voc/label.csv' \
--test_label_dir='dataset/OPA/new_OPA/new_alt_test_label.csv' \
--model='b_vit' \
--loss='mse' \
--mask_dir='aaa' \
--batch_size="32" \
--obj_dir='dataset/final_voc/obj_max' \
--lr='0.0025' --optim='adam' --epoch='50')
