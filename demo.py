import argparse
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb
from os import listdir
from os.path import join, isfile
import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('epoch-5.pth.tar')['state_dict'])
    model.eval()
    mypath='data/demo/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for f in files:
        nyu2_loader = loaddata.readNyu2(join(mypath, f))
        test(nyu2_loader, model, f)


def test(nyu2_loader, model, fl):
    for i, image in enumerate(nyu2_loader):     
        image = torch.autograd.Variable(image, volatile=True).cuda()
        out = model(image)
        
        matplotlib.image.imsave('data/demo/out-'+str(fl)+'.png', out.view(out.size(2),out.size(3)).data.cpu().numpy())

if __name__ == '__main__':
    main()
