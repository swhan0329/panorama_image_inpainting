from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from model.PInet import *
from data.dataset import *
from utils.util import *
from utils.cube_to_equi import c2e
from model.losses import *

import matplotlib.pyplot as plt

from torchvision import transforms

def train(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue
    data_parallel = args.data_parallel

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    network = args.network

    if torch.cuda.is_available() == False:
        raise Exception('At least one gpu must be available.')
    else:
        gpu = torch.device('cuda:0')

    print("mode: %s" % mode)
    print("data_parallel: %s" % data_parallel)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("network: %s" % network)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % gpu)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png'))

    ## 네트워크 학습하기
    if mode == 'train':
        transform_train = transforms.Compose([Resize(shape=(ny, nx)),Normalize(),ToTensor()])

        dataset_train = PanoramaDataset(in_dir=data_dir, transform=transform_train)

        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(loader_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    ## 네트워크 생성하기
    if network == "PInet":
        netG = Generator(pano_in_channels=4, cube_in_channels=4, pano_out_channels=512, cube_out_channels=512, decoder_in_channels=1024, decoder_out_channels=3*6,nker=nker)
        netD = Discriminator(in_channels=nch*6, out_channels=1, nker=nker)

        if data_parallel:
            netG = DataParallel(netG)
            netD = DataParallel(netD)

        netG = netG.to(gpu)
        netD = netD.to(gpu)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    fn_loss = nn.BCELoss().to(gpu)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 1, 3, 4, 2) #0,2,3,1
    fn_denorm = lambda x, mean, std: (x * std) + mean

    cmap = None

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    # writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                                        netG=netG, netD=netD,
                                                        optimG=optimG, optimD=optimD)

        
        for epoch in range(st_epoch + 1, num_epoch + 1):
            netG.train()
            netD.train()

            loss_G_train = []
            loss_D_real_train = []
            loss_D_fake_train = []

            for batch, sample in enumerate(loader_train, 1):
                # forward pass
                cube = sample['cube'].to(gpu, dtype=torch.float32) # B, F, 3, H, W
                cube_mask = sample['cube_mask'].to(gpu, dtype=torch.float32) # B, F, 1, H, W
                pano = sample['pano'].to(gpu, dtype=torch.float32) # B, 1, 3, H, W
                pano_mask = sample['pano_mask'].to(gpu, dtype=torch.float32) # B, 1, 1, H, W

                for f in range(6):
                    x_cube_mask_temp = cube[:, f, :, :, :] - cube[:, f, :, :, :] * cube_mask[:, f, :, :, :]
                    x_cube_mask_temp = x_cube_mask_temp.view(x_cube_mask_temp.shape[0], 1, x_cube_mask_temp.shape[1],
                                                        x_cube_mask_temp.shape[2], x_cube_mask_temp.shape[3])
                    if f == 0:
                        x_cube_mask = x_cube_mask_temp
                    else:
                        x_cube_mask = torch.cat((x_cube_mask,x_cube_mask_temp),dim=1)

                x_pano_mask = pano - pano * pano_mask

                for f in range(6):
                    input_cube_temp = torch.cat((x_cube_mask[:,f,:,:],cube_mask[:,f,:,:]),dim=1)
                    if f == 0:
                        input_cube = input_cube_temp
                    else:
                        input_cube = torch.cat((input_cube,input_cube_temp),dim=1)
                
                input_pano = torch.cat((x_pano_mask, pano_mask), dim=2)
                input_pano = torch.squeeze(input_pano,dim=1)

                input_cube = input_cube.to(dtype=torch.float32)
                input_pano = input_pano.to(dtype=torch.float32)

                _,_,_,output = netG(input_pano,input_cube)

                # backward netD
                set_requires_grad(netD, True)
                optimD.zero_grad()

                for f in range(6):
                    label_cube_temp = cube[:,f,:,:]
                    if f == 0:
                        label_cube = label_cube_temp
                    else:
                        label_cube = torch.cat((label_cube,label_cube_temp),dim=1)

                label_cube = label_cube.to(dtype=torch.float32)

                pred_real = netD(label_cube)
                pred_fake = netD(output.detach())

                loss_D_real = fn_loss(pred_real, torch.ones_like(pred_real))
                loss_D_fake = fn_loss(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad(netD, False)
                optimG.zero_grad()

                pred_fake = netD(output)

                output = torch.reshape(output, (output.shape[0], 6, nch, output.shape[2], output.shape[3]))
                loss_G = (fn_loss(pred_fake, torch.ones_like(pred_fake)) + completion_network_loss(cube, output, cube_mask))/2.

                loss_G.backward()
                optimG.step()

                # 손실함수 계산
                loss_G_train += [loss_G.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_fake_train += [loss_D_fake.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "GEN %.4f | DISC REAL: %.4f | DISC FAKE: %.4f" %
                      (epoch, num_epoch, batch, num_batch_train*batch_size,
                       np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

                for f in range(6):
                    completed_temp = poisson_blend(cube[:,f,:,:,:], output[:,f,:,:,:], cube_mask[:,f,:,:,:])
                    if f == 0:
                        completed = completed_temp
                    else:
                        completed = torch.cat((completed,completed_temp),dim=1)

                completed = torch.reshape(completed, (completed.shape[0], 6, nch, completed.shape[2], completed.shape[3]))

                if batch % 4 == 0:
                    # Tensorboard 저장하기
                    id = num_batch_train * (epoch - 1) + batch

                    completed = fn_tonumpy(fn_denorm(completed, mean=0.5, std=0.5))
                    cube = fn_tonumpy(fn_denorm(cube, mean=0.5, std=0.5))

                    cube_mask = torch.cat((cube_mask, cube_mask, cube_mask), dim=2)
                    cube_mask = fn_tonumpy(fn_denorm(cube_mask, mean=0.5, std=0.5))

                    x_cube_mask = fn_tonumpy(fn_denorm(x_cube_mask, mean=0.5, std=0.5))
                    x_pano_mask = fn_tonumpy(fn_denorm(x_pano_mask, mean=0.5, std=0.5))

                    equirec = c2e(completed[0], h=270, w=480, cube_format='list')

                    for ff in range(6):
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%05d_input_cube%01d.png' % (id, ff)),
                        #            cube[0][ff, :, :, :], cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%05d_input_cube_mask%01d.png' % (id, ff)),
                        #            cube_mask[0][ff, :, :, :], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_train, 'png', '%05d_cube_mask_%01d.png' % (id, ff)),
                                   x_cube_mask[0][ff, :, :, :], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_train, 'png', '%05d_output_%01d.png' % (id,ff)), completed[0][ff,:,:,:], cmap=cmap)

                        # writer_train.add_image('cube_mask_%01d.png' %ff, x_cube_mask[:, ff, :, :, :],id, dataformats='NHWC')
                        # writer_train.add_image('output_%01d.png' %ff, completed[:,ff,:,:,:], id, dataformats='NHWC')

                    plt.imsave(os.path.join(result_dir_train, 'png', '%05d_pano_mask.png' % (id)),
                               x_pano_mask[0, 0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%05d_equi.png' % (id)),
                               equirec, cmap=cmap)
                    writer_train.add_image('pano_mask.png', x_pano_mask[:, 0, :, :, :], id, dataformats='NHWC')
                    writer_train.add_image('equi.png', equirec, id, dataformats='HWC')

            writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

            if epoch % 20 == 0 or epoch == num_epoch:
                parallel_save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

        writer_train.close()

def test(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue
    data_parallel = args.data_parallel
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    network = args.network
    learning_type = args.learning_type

    if torch.cuda.is_available() == False:
        raise Exception('At least one gpu must be available.')
    else:
        gpu = torch.device('cuda:0')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % gpu)

    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png'))
        os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 네트워크 학습하기
    if mode == "test":
        transform_test =transforms.Compose([Resize(shape=(ny, nx, nch)),Normalize(),ToTensor()])

        dataset_test = PanoramaDataset(in_dir=data_dir, transform=transform_test)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    ## 네트워크 생성하기
    if network == "PInet":
        netG = Generator(pano_in_channels=nch, cube_in_channels=nch,pano_out_channels=nch,cube_out_channels=nch, nker=nker).to(gpu)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(gpu)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

        if data_parallel:
            netG = DataParallel(netG)
            netD = DataParallel(netD)

        netG = netG.to(gpu)
        netD = netD.to(gpu)

    ## 손실함수 정의하기
    fn_loss = nn.BCELoss().to(gpu)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = None

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == "test":
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

        with torch.no_grad():
            netG.eval()

            input = torch.randn(batch_size, 100, 1, 1).to(gpu)
            output = netG(input)

            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(output.shape[0]):
                id = j

                output_ = output[j]
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                output_ = np.clip(output_, a_min=0, a_max=1)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)