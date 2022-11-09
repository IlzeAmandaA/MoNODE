import os 

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from odevae import ODEVAE
from invodevae import INVODEVAE
from utilities import plot_rot_mnist, get_dataset, plot_latent_traj

num_workers = 0
Nepoch = 500
solver = 'rk4'

trainset, testset = get_dataset(num_workers=0, batch_size=25, N=50, augment_data=True, device=device)
odevae = ODEVAE(order=1, q=4, n_filt=8, is_mlp=False).to(device)
# odevae = INVODEVAE(order=1, q=2, n_filt=4, qs=5, Hf=150, actf='relu').to(device)

odevae.load_state_dict(torch.load('etc/odevae_mnist.pth'))
odevae.eval()

# for i,local_batch in enumerate(trainset):
#     train_batch = local_batch.to(device)
#     Xrec, qz0_m, qz0_v, zt, lhood, kl_z = odevae(train_batch, method='rk4', Tode=2*train_batch.shape[1])
#     plot_latent_traj(zt)
#     plot_rot_mnist(train_batch, Xrec)
#     print(ahahaha)

# z = odevae.fc1(odevae.encoder(tr_batch.reshape(25*16,1,28,28))).reshape(25,16,2).detach().cpu()
# plt.figure(1,(6,6))
# for i in range(4):
#     plt.plot(z[i,:,0], z[i,:,1], '*-');

# plt.savefig('figs/enc.png',dpi=200)
# plt.close()


optimizer = torch.optim.Adam(odevae.parameters(),lr=5e-3)
for ep in range(Nepoch):
    for i,local_batch in enumerate(trainset):
        optimizer.zero_grad()
        tr_batch = local_batch.to(device)
        Xrec_tr, qz0_m, qz0_v, zt_tr, lhood, kl_z = odevae(tr_batch, len(trainset), solver=solver)
        tr_loss = -lhood + kl_z
        tr_loss.backward()
        optimizer.step()
        # print('Iter:{:<2d} lhood:{:8.2f}  kl_z:{:<8.2f} '.format(i, lhood.item(), kl_z.item()))
        if not odevae.is_mlp:
            odevae.f.fix_gpytorch_cache(i)
    with torch.no_grad():
        Xrec_tr, qz0_m, qz0_v, zt_tr, lhood, kl_z = odevae(tr_batch, len(trainset), solver=solver, Tode=2*tr_batch.shape[1])
        for test_batch in testset:
            test_batch = test_batch.to(device)
            Xrec_test, qz0_m, qz0_v, zt_test, lhood, kl_z = odevae(test_batch, len(test_batch), solver=solver, Tode=2*test_batch.shape[1])
            test_loss = -lhood + kl_z
            break
    print('Epoch:{:4d}/{:4d} tr_loss:{:8.2f}  test_loss:{:5.3f}'.format(ep, Nepoch, tr_loss.item(), test_loss.item()))
    if ep%10==0:
        plot_rot_mnist(tr_batch,   Xrec_tr,   fname='rot_mnist_tr.png')
        plot_rot_mnist(test_batch, Xrec_test, fname='rot_mnist_test.png')
        plot_latent_traj(zt_tr,   fname='rot_mnist_latents_tr.png')
        plot_latent_traj(zt_test, fname='rot_mnist_latents_test.png')
        # torch.save(odevae.state_dict(), os.path.join('etc','odevae_mnist.pth'))

