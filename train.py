import fire 
import os
import torch
from utils import *
import json

gpu = torch.device('cuda:0')

def get_dataset(dataloader, img_dir, label_dir, obj_dir, model):
    match model:
        case "double_resnet50":
            transform = resnet_preprocess
    
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

def get_model(model):
    match model:
        case "double_resnet50":
            return DoubleResnet50()
        
def get_loss(loss):
    match loss:
        case "mse":
            return torch.nn.MSELoss(reduction='sum')
        case "var_mse_min":
            return var_mse_min
        
def get_optim(optim, model, lr):
    match optim:
        case "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)


def main(dataloader, img_dir, label_dir, model, loss,
         lr, optim, epoch, test_label_dir=None,
         batch_size="16", obj_dir=None):
    dataloaders = ["one_channel", "two_channel", "OPA"]
    models = ["double_resnet50"]
    losses = ["mse", "var_mse_min"]
    optims = ["adam"]

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
        train_dataset = get_dataset(dataloader, img_dir, label_dir, obj_dir, model)
        test_dataset = get_dataset(dataloader, img_dir, test_label_dir, obj_dir, model)
    else:
        dataset = get_dataset(dataloader, img_dir, label_dir, obj_dir, model)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    model = get_model(model)
    model.to(gpu)
    loss = get_loss(loss)
    optim = get_optim(optim,model,lr)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    result_dict = {}

    for t in range(epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss, optim, result_dict, t)
        test_loop(test_dataloader, model, loss, result_dict, t)
        print("-------------------------------\n")
    print(("*"*100)+"\nDone!")

    with open("output/trainingresult.json", "w") as f:
        json.dump(result_dict, f)
    print("Training result saved to output/trainingresult.json")

    torch.save(model.state_dict(), "weight.pt")
    print("Model weight saved to weight.pt")

    print("Saving result on test set. This will take a while...")
    save_eval(test_dataloader, model, loss)
    print("ALL DONE!")


    



if __name__ == "__main__":
    print(f"cuda is available: {torch.cuda.is_available()}")
    print(f"current device: {torch.cuda.current_device()}")
    print(f"using {gpu}")
    fire.Fire(main)