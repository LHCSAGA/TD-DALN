import torch
import train
import model
import data_load



def main():
    source_train_loader = data_load.source_train_loader
    target_train_loader = data_load.target_train_loader
    if torch.cuda.is_available():
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()
        train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader)
    else:
        print("No GPUs available.")

if __name__ == "__main__":
    main()
