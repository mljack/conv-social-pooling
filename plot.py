from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True    # visualize multi-model predictions
args['train_flag'] = False

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/cslstm_m.tar'))
#net.load_state_dict(torch.load('trained_models/cslstm.tar'))
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset('data/TestSet.mat')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=False,num_workers=1,collate_fn=tsSet.collate_fn)

skip = 10

lossVals = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()

delta = 1.0
x_min = -120
x_max = 120
range_y = 20
scale = 0.4
v_w = 5
v_l = 15

for i, data in enumerate(tsDataloader):
    if skip > 0:
        skip -= 1
        continue

    st_time = time.time()
    hist, nbrs, nbrs_fur, mask, lat_enc, lon_enc, fut, op_mask = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()

    # Forward pass
    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
    nbr_count_all = 0
    for k in range(lat_pred.shape[0]):  # batch
        fig, ax = plt.subplots(figsize=((x_max-x_min)*scale,range_y*2*scale))
        Z = None

        #print(hist.shape)   # [16, 128, 2]
        #print(nbrs.shape)   # [16, 939, 2]
        #print(mask.shape)   # [128, 3, 13, 64]
        #print(np.count_nonzero(mask)/nbrs.shape[1]) # 64
        #print(mask[0, :, :, 0])
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        # [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        nbr_count = np.count_nonzero(mask[k, :, :, 0])
        #print(nbr_count)    # 7 neighbors
        local_nbrs = nbrs[:, nbr_count_all:nbr_count_all+nbr_count, :]
        nbr_count_all += nbr_count
        #print(local_nbrs.shape) # [16, 7, 2]    found 7 neighbor in 3*13 grid
        #print(lat_enc.shape)    # [128, 3]
        #print(lon_enc.shape)    # [128, 2]
        #print(len(nbrs_fur))   # 128 grids with neighbor future

        for nbr_fur in nbrs_fur[k]:
            if nbr_fur.size > 0:
                nbr_fur_x = nbr_fur[:,0]
                nbr_fur_y = nbr_fur[:,1]
                plt.plot(nbr_fur_y, nbr_fur_x, '|-')    # neighbor future

        if local_nbrs.nelement() > 0:
            for j in range(local_nbrs.shape[1]):
                nbr_x = local_nbrs[:,j,0].cpu().numpy()
                nbr_y = local_nbrs[:,j,1].cpu().numpy()
                #plt.plot(nbr_y, nbr_x, 'b|-')  # neighbor history
                patch = matplotlib.patches.Rectangle((nbr_y[-1]-v_l*0.5, nbr_x[-1]-v_w*0.5), v_l, v_w, color="w", fill=False)
                ax.add_patch(patch) # neighbor vehicles

        hist_x = hist[:,k,0].cpu().numpy()
        hist_y = hist[:,k,1].cpu().numpy()
        plt.plot(hist_y, hist_x, 'w|-')

        fur_x = fut[:,k,0].cpu().numpy()
        fur_y = fut[:,k,1].cpu().numpy()
        plt.plot(fur_y, fur_x, 'w|-')

        lat_man = torch.argmax(lat_pred[k, :]).detach()
        lon_man = torch.argmax(lon_pred[k, :]).detach()
        selected_indx = lon_man*3 + lat_man
        man_count = 0
        for j in range(lat_pred.shape[1]):  # lat
            for jj in range(lon_pred.shape[1]): # lon
                lat_man = j
                lon_man = jj
                indx = lon_man*3 + lat_man

                prob = lat_pred[k, lat_man].detach() * lon_pred[k, lon_man].detach()
                if prob < 0.2:
                    continue
                man_count += 1

                #lon_man = lon_man.cpu().numpy()
                #lat_man = lat_man.cpu().numpy()
                #if lat_man == 0:
                #    continue
                #print("\nlon: ", lon_man, "\nlat: ", lat_man)

                fut_pred_ = fut_pred[indx][:,k,:].cpu().detach().numpy()

                x = np.arange(x_min, x_max, delta)
                y = np.arange(-range_y, range_y, delta)
                X, Y = np.meshgrid(x, y)
                xx = []
                yy = []

                for i in range(fut_pred_.shape[0]): # time
                    #print(fut_pred_.shape)
                    muX = fut_pred_[i,0]
                    muY = fut_pred_[i,1]
                    sigX = fut_pred_[i,2]
                    sigY = fut_pred_[i,3]
                    rho = fut_pred_[i,4]
                    #print(muX, muY, sigX, sigY, rho)

                    xx.append(muX)
                    yy.append(muY)

                    Z1 = mlab.bivariate_normal(X, Y, 1/sigY, 1/sigX, muY, muX, rho/sigX/sigY)

                    factor = (Z1.max() - Z1.min())
                    if factor < 0.00001:
                        continue
                    Z1 /= factor
                    Z1 *= prob

                    #Z1 = np.log(Z1+1)
                    #print("range:", np.min(Z1), np.max(Z1))
                    if Z is None:
                        Z = Z1.copy()
                    else:
                        Z = Z + Z1
                    #if i >= 5:
                    #    break

                if indx == selected_indx:
                    plt.plot(yy, xx, 'rx-') # maneuver with max prob
                else:
                    plt.plot(yy, xx, 'y-')  # other maneuvers
        if Z is not None: # and man_count > 1:
            patch = matplotlib.patches.Rectangle((-v_l*0.5, -v_w*0.5), v_l, v_w, color="y", fill=False)
            ax.add_patch(patch) # main vehicle
            m = plt.imshow(Z, interpolation='bilinear', origin='lower', cmap=cm.inferno, extent=(x_min, x_max, -range_y, range_y))
            plt.show()
        else:
            plt.clf()
            plt.close()


