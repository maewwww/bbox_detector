import fire 
import os
import torch
from utils import *
import json

gpu = torch.device('cuda:0')

def get_dataset(dataloader, img_dir, label_dir, obj_dir, model, mask_dir):
    match model:
        case "double_resnet50":
            transform = resnet_preprocess
        case "lraspp":
            transform = LRASPP_preprocess
            target_transform = LRASPP_target_preprocess
        case "unet":
            transform = unet_preprocess
            target_transform = unet_target_preprocess
        case "ynet":
            transform = unet_preprocess
            target_transform = unet_target_preprocess
        case "b_vit":
            transform = vit_transform
            target_transform = None
        case "fusion_2":
            transform = vit_transform_2
            obj_transform = vit_transform_2 #efficient_net_transform
            target_transform = None
        case "fusion_3":
            transform = vit_transform
            obj_transform = vit_transform_2
            target_transform = None

    
    match dataloader:
        case "one_channel":
            raise NotImplemented
        case "two_channel":
            return TwoChannelCustomDataset(label_dir=label_dir, img_dir=img_dir, obj_dir=obj_dir,
                                           transform=transform,
                                           target_transform=None,)
        case "OPA":
            return OPADataset(label_dir=label_dir, img_dir=img_dir, obj_dir=obj_dir,
                                           transform=transform,
                                           target_transform=None)
        case "OPA_Dist":
            return OPADistDataset(label_dir=label_dir, img_dir=img_dir, obj_dir=obj_dir,
                                  mask_dir=mask_dir, transform=transform, target_transform=target_transform)
        case "OPA_2":
            return OPADataset_2(label_dir=label_dir, img_dir=img_dir, obj_dir=obj_dir,
                                  transform=transform, target_transform=target_transform, obj_transform=obj_transform)

def get_model(model, batch_size=8):
    match model:
        case "double_resnet50":
            return DoubleResnet50()
        case "lraspp":
            return OneOutLRASPP()
        case "unet":
            return UNet(n_channels=6, n_classes=1)
        case "ynet":
            return YNet(bg_channels=3, obj_channels=3, n_classes=1, bilinear=False)
        case "b_vit":
            return base_vit_6_channels()
        case "fusion_2":
            return fusion_v2()
        case "fusion_3":
            return fusion_v3(55, batch_size)
def get_loss(loss):
    match loss:
        case "mse":
            return torch.nn.MSELoss(reduction='sum')
        case "var_mse_min":
            return var_mse_min
        case "kldiv":
            return kldiv
        case "dice":
            return dice_loss
        case "matching":
            return matching_loss
        
def get_optim(optim, model, lr):
    match optim:
        case "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)


def main(dataloader, img_dir, label_dir, mask_dir, model, loss,
         lr, optim, epoch, test_label_dir=None,
         batch_size="16", obj_dir=None):
    dataloaders = ["one_channel", "two_channel", "OPA", "OPA_Dist", "OPA_2"]
    models = ["double_resnet50", "lraspp", "unet", "ynet", "b_vit", "fusion_2", "fusion_3"]
    losses = ["mse", "var_mse_min", "kldiv", "dice", "matching"]
    optims = ["adam"]
    #torch.multiprocessing.set_start_method('spawn')
    #torch.autograd.set_detect_anomaly(True)

    assert dataloader in dataloaders, "Unknown Dataloader"
    print(f"{dataloader} is chosen as dataloader")

    assert model in models, "Unknown Model"
    print(f"{model} is chosen as model")

    assert str(batch_size).isdecimal(), "Invalid batch_size"
    print(f"{batch_size} is chosen as batch_size")

    assert loss in losses, "Unknown loss function"
    print(f"{loss} is chosen as loss function")

    assert optim in optims, "Unknown optimizer"
    print(f"{optim} is chosen as optimizer")

    assert os.path.exists(img_dir), "img dir doesn't exist"
    print(f"{img_dir} is chosen as img_dir")

    assert os.path.exists(label_dir), "label dir doesn't exist"
    print(f"{label_dir} is chosen as label_dir")

    if obj_dir is not None:
        assert os.path.exists(obj_dir), "obj dir doesn't exist"
        print(f"{obj_dir} is chosen as obj_dir")
    else:
        print("obj_dir is NONE!")

    lr, batch_size, epoch = float(lr), int(batch_size), int(epoch)

    if test_label_dir is not None:
        train_dataset = get_dataset(dataloader, img_dir, label_dir, obj_dir, model, mask_dir)
        test_dataset = get_dataset(dataloader, img_dir, test_label_dir, obj_dir, model, mask_dir)
    else:
        dataset = get_dataset(dataloader, img_dir, label_dir, obj_dir, model)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    model = get_model(model, batch_size)
    #model.load_state_dict(torch.load("checkpoint.pt", weights_only=True))
    model.to(gpu)
    print("parameters:", sum(p.numel() for p in model.parameters())) 
    #print("trainable", pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)) 
    loss = get_loss(loss)
    optim = get_optim(optim,model,lr)
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    result_dict = {}

    for t in range(epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss, optim, result_dict, t)
        test_loop(test_dataloader, model, loss, result_dict, t)
        print("-------------------------------\n")

        # save intermediate model every 10 epochs
        if t != 1 and t % 3 == 0:
            i_modelname = str(t + 1) + "weight.pt"
            torch.save(model.state_dict(), i_modelname)
            print("Model weight saved to ", i_modelname)

    print(("*"*100)+"\nDone!")

    if not os.path.exists("output/"):
        os.makedirs("output/")

    with open("output/trainingresult.json", "w") as f:
        json.dump(result_dict, f)
    print("Training result saved to output/trainingresult.json")

    torch.save(model.state_dict(), "weight.pt")
    print("Model weight saved to weight.pt")

    print("Saving result on test set.")
    save_eval(test_dataloader, model, loss)
    #save_dist_eval(test_dataloader,model,kldiv,out_dir="local_output")
    print("ALL DONE!")


    



if __name__ == "__main__":
    print(f"cuda is available: {torch.cuda.is_available()}")
    print(f"current device: {torch.cuda.current_device()}")
    fire.Fire(main)