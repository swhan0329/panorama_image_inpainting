from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from model.PIUnet2 import *
from data.dataset import *
from pre_proc.create_data import *
from utils.util import *
from utils.cube_to_equi import c2e

import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from torch.utils.data import DataLoader


def train(args):
    torch.manual_seed(2020)
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

    nker = args.nker
    norm = args.norm

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
    print("norm: %s" % norm)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % gpu)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')
    result_dir_val = os.path.join(result_dir, 'val')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png'))
    if not os.path.exists(result_dir_val):
        os.makedirs(os.path.join(result_dir_val, 'png'))

    ## 네트워크 학습하기
    if mode == 'train':
        transform_train = transforms.Compose([Normalize(), ToTensor()])
        transform_val = transforms.Compose([Normalize(), ToTensor()])

        dataset_train = PanoramaDataset(in_dir=os.path.join(
            data_dir, 'train'), transform=transform_train)
        dataset_val = PanoramaDataset(in_dir=os.path.join(
            data_dir, 'val'), transform=transform_val)

        loader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        loader_val = DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(loader_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

        num_data_val = len(loader_val)
        num_batch_val = np.ceil(num_data_val / batch_size)

    ## 네트워크 생성하기
    if network == "PIUnet":
        netFaceG = FaceGenerator(
            in_channels=4, out_channels=3, nker=nker, norm=norm, relu=True)
        netFaceD = FaceDis(in_channels=6, out_channels=1,
                           nker=nker, norm=norm, relu=True)
        netCubeG = CubeGenerator(
            in_channels=4, out_channels=3, nker=nker, norm=norm, relu=True)
        netWholeD = WholeDis(in_channels=6 * 6, nker=nker,
                             norm=norm, relu=True)
        netSliceD = SliceDis(in_channels=6, out_channels=1,
                             nker=nker, norm=norm, relu=True)

        if data_parallel:
            netFaceG = DataParallel(netFaceG)
            netFaceD = DataParallel(netFaceD)
            netCubeG = DataParallel(netCubeG)
            netWholeD = DataParallel(netWholeD)
            netSliceD = DataParallel(netSliceD)

        netFaceG = netFaceG.to(gpu)
        netFaceD = netFaceD.to(gpu)
        netCubeG = netCubeG.to(gpu)
        netWholeD = netWholeD.to(gpu)
        netSliceD = netSliceD.to(gpu)

    ## 손실함수 정의하기
    fn_l1 = nn.L1Loss().to(gpu)
    fn_gan = nn.BCELoss().to(gpu)

    ## Optimizer 설정하기  
    optimFG = torch.optim.Adam(netFaceG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimCG = torch.optim.Adam(netCubeG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimFD = torch.optim.Adam(netFaceD.parameters(), lr=lr, betas=(0.5, 0.999))
    d_params = list(netWholeD.parameters()) + list(netSliceD.parameters())
    optimCD = torch.optim.Adam(d_params, lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to(
        'cpu').detach().numpy().transpose(0, 1, 3, 4, 2)
    fn_tonumpy_4 = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean

    cmap = None

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    ## 네트워크 학습시키기
    st_epoch = 0
    FG = sum(np.prod(list(p.size())) for p in netFaceG.parameters())
    CG = sum(np.prod(list(p.size())) for p in netCubeG.parameters())
    FD = sum(np.prod(list(p.size())) for p in netFaceD.parameters())
    WD = sum(np.prod(list(p.size())) for p in netWholeD.parameters())
    SD = sum(np.prod(list(p.size())) for p in netSliceD.parameters())

    print('Number of params in netFaceG: %d' % FG)
    print('Number of params in netFD: %d' % FD)
    print('Number of params in netCubeG: %d' % CG)
    print('Number of params in netWD: %d' % WD)
    print('Number of params in netSD: %d' % SD)
    print('Total Number of params in Network: %d' % (FG+FD+CG+WD+SD))

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netFaceG, netFaceD, optimFG, optimFD, st_epoch = face_load(ckpt_dir=ckpt_dir,
                                                                      netFaceG=netFaceG, netFaceD=netFaceD, optimFG=optimFG, optimFD=optimFD)
            netCubeG, netWholeD, netSliceD, optimCG, optimCD, st_epoch = cube_load(ckpt_dir=ckpt_dir,
                                                                      netCubeG=netCubeG, netWholeD=netWholeD,
                                                                       netSliceD=netSliceD, optimCG=optimCG, optimCD=optimCD)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netFaceG.train()
            netFaceD.train()
            netCubeG.train()
            netWholeD.train()
            netSliceD.train()

            loss_FG_train = []
            loss_CG_train = []

            loss_FD_train = []
            loss_CD_train = []

            for batch, sample in enumerate(loader_train, 1):
                # forward pass
                cube = sample['cube'].to(
                    gpu, dtype=torch.float32)  # B, F, 3, H, W
                cube_mask = sample['cube_mask'].to(
                    gpu, dtype=torch.float32)  # B, F, 1, H, W
                cube_mask_4 = cube_mask.view(
                    cube_mask.shape[0], cube_mask.shape[1], cube_mask.shape[3], cube_mask.shape[4])

                # face order is ['f', 'r', 'b', 'l', 't', 'd']
                # we need 4 faces like this order ['f', 'r', 'b', 'l']
                # concate 4 faces
                # ground truth 4 faces -> B, 3, H, W*4
                g4f = torch.cat(
                    (cube[:, 0], torch.flip(cube[:, 1], [3]), torch.flip(cube[:, 2], [3]), cube[:, 3]), dim=3)
                # mask 4 faces -> B, 1, H, W*4
                m4f = torch.cat(
                    (cube_mask[:, 0], torch.flip(cube_mask[:, 1], [3]), torch.flip(cube_mask[:, 2], [3]),
                     cube_mask[:, 3]),
                    dim=3)
                # with mask 4 faces -> B, 3, H, W*4
                cm4f = g4f - g4f * m4f

                # st1_output -> BN, C, H, W*4
                # st2_output -> BN, 6*C, H, W
                st1_output = netFaceG(g4f, m4f, cm4f)
                st1_output_inp = st1_output*m4f + g4f*(1-m4f)

                # st1_output -> BN, 4, C, H, W
                st1_output_5 = st1_output.view(st1_output.shape[0], 4, st1_output.shape[1], st1_output.shape[2],
                                                    int(st1_output.shape[3] / 4))

                st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3 = torch.split(
                    st1_output_5, 1, dim=1)

                st1_output_split1 = torch.flip(st1_output_split1[:], [2])
                st1_output_split2 = torch.flip(st1_output_split2[:], [2])
                st1_output_5 = torch.cat(
                    (st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3), dim=1)

                # st1_cube -> BN, 6, C, H, W
                # st1_cube_4 -> BN, 6*C, H, W
                st1_cube = torch.cat(
                    (st1_output_5, cube[:, 4:6]*(1-cube_mask[:, 4:6])), dim=1)
                st1_cube_inp = st1_cube*cube_mask+cube*(1-cube_mask)

                st1_cube_inp_4 = st1_cube_inp.view(
                    (st1_cube_inp.shape[0], -1, st1_cube_inp.shape[3], st1_cube_inp.shape[4]))

                cube_mask_4 = cube_mask.view(
                    cube_mask.shape[0], cube_mask.shape[1], cube_mask.shape[3], cube_mask.shape[4])
                st2_input = torch.cat((st1_cube_inp_4, cube_mask_4), dim=1)

                st2_output = netCubeG(st2_input)
                st2_output_5 = st2_output.view(
                    st2_output.shape[0], 6, 3, st2_output.shape[2], st2_output.shape[3])

                # cube + cube_mask
                # x_cube_mask_5 -> BN, 6, 3, H, W
                x_cube_mask_5 = cube - cube*cube_mask
                
                # inpainted_cube -> BN, 6, 3, H, W
                inpainted_cube = st2_output_5*cube_mask+(1-cube_mask)*cube
            
                # input cube + cube mask / (net output + cube) + cube mask
                # x_real_cube_mask -> BN, 6*6, H, W
                # x_fake_cube_mask -> BN, 6*6, H, W
                x_real_cube_mask = torch.cat((st1_cube_inp, cube), dim=2)
                x_fake_cube_mask = torch.cat((st1_cube_inp, st2_output_5), dim=2)
                x_real_cube_mask = x_real_cube_mask.view(
                    x_real_cube_mask.shape[0], -1, x_real_cube_mask.shape[3], x_real_cube_mask.shape[4])
                x_fake_cube_mask = x_fake_cube_mask.view(
                    x_fake_cube_mask.shape[0], -1, x_fake_cube_mask.shape[3], x_fake_cube_mask.shape[4])

                # backward netWholeD, netSliceD
                set_requires_grad(netWholeD, True)
                set_requires_grad(netSliceD, True)
                optimCD.zero_grad()

                pred_whole_real = netWholeD(x_real_cube_mask)
                pred_whole_fake = netWholeD(x_fake_cube_mask.detach())

                pred_slice_real = netSliceD(x_real_cube_mask)
                pred_slice_fake = netSliceD(x_fake_cube_mask.detach())

                whole_penalty = calc_gradient_penalty(netWholeD, x_real_cube_mask, x_fake_cube_mask.detach(), gpu)
                slice_penalty = calc_gradient_penalty(netSliceD, x_real_cube_mask, x_fake_cube_mask.detach(), gpu)
                loss_wgan_gp = whole_penalty + slice_penalty
                loss_wgan_d = torch.mean(pred_whole_fake - pred_whole_real) + torch.mean(pred_slice_fake - pred_slice_real)
                loss_CD = loss_wgan_gp * 10 + loss_wgan_d

                loss_CD.backward(retain_graph=True)

                # backward netCubeG
                set_requires_grad(netWholeD, False)
                set_requires_grad(netSliceD, False)
                optimCG.zero_grad()
    
                pred_whole_fake = netWholeD(x_fake_cube_mask)
                pred_slice_fake = netSliceD(x_fake_cube_mask)

                loss_l1 = fn_l1(st2_output_5*cube_mask, cube*cube_mask)
                loss_ae = fn_l1(st2_output_5*(1-cube_mask),cube*(1-cube_mask))
                loss_G_wgan = - torch.mean(pred_whole_fake) - torch.mean(pred_slice_fake)
                loss_CG = loss_G_wgan*0.001 + loss_l1 * 10 + loss_ae
                
                loss_CG.backward(retain_graph=True)
                
                # backward netFaceD
                set_requires_grad(netFaceD, True)
                optimFD.zero_grad()

                # input face + face gt / net output + face gt
                # x_real_face_mask -> BN, 6, H, W*4
                # x_fake_face_mask -> BN, 6, H, W*4
                x_real_face_mask = torch.cat((cm4f, g4f), dim=1)
                x_fake_face_mask = torch.cat((cm4f, st1_output), dim=1)

                pred_face_real = netFaceD(x_real_face_mask)
                pred_face_fake = netFaceD(x_fake_face_mask.detach())
                dis_real_loss = fn_gan(
                    pred_face_real, torch.ones_like(pred_face_real))
                dis_fake_loss = fn_gan(
                    pred_face_fake, torch.zeros_like(pred_face_fake))
                loss_FD = (dis_real_loss + dis_fake_loss) / 2 * 100

                loss_FD.backward(retain_graph=True)

                # backward netFaceG
                set_requires_grad(netFaceD, False)
                optimFG.zero_grad()

                pred_face_fake = netFaceD(x_fake_face_mask)

                loss_l1 = fn_l1(st1_output * m4f, g4f * m4f)
                loss_ae = fn_l1(st1_output*(1-m4f), g4f*(1-m4f))
                loss_FG_gan = fn_gan(pred_face_fake, torch.ones_like(pred_face_fake))
                loss_FG = loss_FG_gan*0.001 + loss_l1 * 10 + loss_ae
                
                loss_FG.backward()

                optimCD.step()
                optimCG.step()
                optimFD.step()
                optimFG.step()        

                #lr_scheduler_CD.step() 
                #lr_scheduler_CG.step() 
                #lr_scheduler_FD.step()  
                #lr_scheduler_FG.step()         
              
                # 손실함수 계산
                loss_FG_train += [loss_FG.item()]
                loss_CG_train += [loss_CG.item()]
                loss_FD_train += [loss_FD.item()]
                loss_CD_train += [loss_CD.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "FG %.4f | CG %.4f | FD %.4f | CD %.4f" %
                      (epoch, num_epoch, batch, num_batch_train * batch_size,
                       np.mean(loss_FG_train), np.mean(loss_CG_train),
                       np.mean(loss_FD_train), np.mean(loss_CD_train)))

                if batch % 30 == 0:
                    # Tensorboard 저장하기
                    id = num_batch_train * (epoch - 1) + batch

                    # 4 face ori
                    g4f = fn_tonumpy_4(fn_denorm(g4f, mean=0.5, std=0.5))
                    # 4 face with mask
                    mask4f = fn_tonumpy_4(fn_denorm(cm4f, mean=0.5, std=0.5))
                    # 4 face inpaint result
                    result4f = fn_tonumpy_4(
                        fn_denorm(st1_output_inp, mean=0.5, std=0.5))

                    inpainted_cube = fn_tonumpy(
                        fn_denorm(inpainted_cube, mean=0.5, std=0.5))
                    cube = fn_tonumpy(fn_denorm(cube, mean=0.5, std=0.5))
                    x_cube_mask_5 = fn_tonumpy(
                        fn_denorm(x_cube_mask_5, mean=0.5, std=0.5))

                    equirec_ori = c2e(
                        cube[0], h=256, w=512, cube_format='list')
                    equirec_ori_mask = c2e(
                        x_cube_mask_5[0], h=256, w=512, cube_format='list')
                    equirec = c2e(
                        inpainted_cube[0], h=256, w=512, cube_format='list')

                    plt.imsave(os.path.join(result_dir_train, 'png', '%07d_4face_mask.png' % (id)),
                               mask4f[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%07d_4face.png' % (id)),
                               result4f[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%07d_4face_ori.png' % (id)),
                               g4f[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%07d_pano_mask.png' % (id)),
                               equirec_ori_mask, cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%07d_pano.png' % (id)),
                               equirec, cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%07d_pano_ori.png' % (id)),
                               equirec_ori, cmap=cmap)

                    writer_train.add_image(
                        '4face_mask.png', mask4f[0], id, dataformats='HWC')
                    writer_train.add_image(
                        '4face.png', result4f[0], id, dataformats='HWC')
                    writer_train.add_image(
                        'pano_ori.png', equirec_ori, id, dataformats='HWC')
                    writer_train.add_image(
                        'pano_mask.png', equirec_ori_mask, id, dataformats='HWC')
                    writer_train.add_image(
                        'pano.png', equirec, id, dataformats='HWC')

            writer_train.add_scalar('loss_FG', np.mean(loss_FG_train), epoch)
            writer_train.add_scalar('loss_CG', np.mean(loss_CG_train), epoch)
            writer_train.add_scalar('loss_FD', np.mean(loss_FD_train), epoch)
            writer_train.add_scalar('loss_CD', np.mean(loss_CD_train), epoch)

            with torch.no_grad():
                netFaceG.eval()
                netFaceD.eval()
                netCubeG.eval()
                netWholeD.eval()
                netSliceD.eval()

                loss_FG_val = []
                loss_CG_val = []

                for batch, sample in enumerate(loader_val, 1):
                    # forward pass
                    cube = sample['cube'].to(gpu, dtype=torch.float32)  # B, F, 3, H, W
                    cube_mask = sample['cube_mask'].to(gpu, dtype=torch.float32)  # B, F, 1, H, W
                    cube_mask_4 = cube_mask.view(cube_mask.shape[0],cube_mask.shape[1],cube_mask.shape[3],cube_mask.shape[4])

                    # face order is ['f', 'r', 'b', 'l', 't', 'd']
                    # we need 4 faces like this order ['f', 'r', 'b', 'l']
                    # concate 4 faces
                    # ground truth 4 faces -> B, 3, H, W*4
                    g4f = torch.cat(
                        (cube[:, 0], torch.flip(cube[:, 1], [3]), torch.flip(cube[:, 2], [3]), cube[:, 3]), dim=3)
                    # mask 4 faces -> B, 1, H, W*4
                    m4f = torch.cat(
                        (cube_mask[:, 0], torch.flip(cube_mask[:, 1], [3]), torch.flip(cube_mask[:, 2], [3]),
                        cube_mask[:, 3]),
                        dim=3)
                    # with mask 4 faces -> B, 3, H, W*4
                    cm4f = g4f - g4f * m4f
                    
                    # st1_output -> BN, C, H, W*4
                    # st2_output -> BN, 6*C, H, W
                    st1_output = netFaceG(g4f, m4f, cm4f)
                    st1_output_inp = st1_output*m4f + g4f*(1-m4f)

                    # st1_output -> BN, 4, C, H, W
                    st1_output_5 = st1_output.view(st1_output.shape[0], 4, st1_output.shape[1], st1_output.shape[2],
                                                        int(st1_output.shape[3] / 4))

                    st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3 = torch.split(
                        st1_output_5, 1, dim=1)

                    st1_output_split1 = torch.flip(st1_output_split1[:], [2])
                    st1_output_split2 = torch.flip(st1_output_split2[:], [2])
                    st1_output_5 = torch.cat(
                        (st1_output_split0, st1_output_split1, st1_output_split2, st1_output_split3), dim=1)

                    # st1_cube -> BN, 6, C, H, W
                    # st1_cube_4 -> BN, 6*C, H, W
                    st1_cube = torch.cat(
                        (st1_output_5, cube[:, 4:6]*(1-cube_mask[:, 4:6])), dim=1)
                    st1_cube_inp = st1_cube*cube_mask+cube*(1-cube_mask)

                    st1_cube_inp_4 = st1_cube_inp.view(
                        (st1_cube_inp.shape[0], -1, st1_cube_inp.shape[3], st1_cube_inp.shape[4]))

                    cube_mask_4 = cube_mask.view(
                        cube_mask.shape[0], cube_mask.shape[1], cube_mask.shape[3], cube_mask.shape[4])
                    st2_input = torch.cat((st1_cube_inp_4, cube_mask_4), dim=1)

                    st2_output = netCubeG(st2_input)
                    st2_output_5 = st2_output.view(
                        st2_output.shape[0], 6, 3, st2_output.shape[2], st2_output.shape[3])

                    # cube + cube_mask
                    # x_cube_mask_5 -> BN, 6, 3, H, W
                    x_cube_mask_5 = cube - cube*cube_mask
                    
                    # inpainted_cube -> BN, 6, 3, H, W
                    inpainted_cube = st2_output_5*cube_mask+(1-cube_mask)*cube
                
                    # input cube + cube mask / (net output + cube) + cube mask
                    # x_real_cube_mask -> BN, 6*6, H, W
                    # x_fake_cube_mask -> BN, 6*6, H, W
                    x_fake_cube_mask = torch.cat((st1_cube_inp, st2_output_5), dim=2)
                    x_fake_cube_mask = x_fake_cube_mask.view(
                        x_fake_cube_mask.shape[0], -1, x_fake_cube_mask.shape[3], x_fake_cube_mask.shape[4])

                    # backward netCubeG
                    pred_whole_fake = netWholeD(x_fake_cube_mask)
                    pred_slice_fake = netSliceD(x_fake_cube_mask)

                    loss_l1 = fn_l1(st2_output_5*cube_mask, cube*cube_mask)
                    loss_ae = fn_l1(st2_output_5*(1-cube_mask), cube*(1-cube_mask))
                    loss_G_wgan = - torch.mean(pred_whole_fake) - torch.mean(pred_slice_fake)
                    loss_CG = loss_G_wgan*0.001 + loss_l1 * 10 + loss_ae

                    # backward netFaceG
                    x_fake_face_mask = torch.cat((cm4f, st1_output), dim=1)
                    pred_face_fake = netFaceD(x_fake_face_mask)

                    loss_l1 = fn_l1(st1_output * m4f, g4f * m4f)
                    loss_ae = fn_l1(st1_output*(1-m4f),g4f*(1-m4f))
                    loss_FG_gan = fn_gan(pred_face_fake, torch.ones_like(pred_face_fake))
                    loss_FG = loss_FG_gan*0.001 + loss_l1 * 10+loss_ae

                    # 손실함수 계산
                    loss_FG_val += [loss_FG.item()]
                    loss_CG_val += [loss_CG.item()]

                    print("VAL: EPOCH %04d / %04d | BATCH %04d / %04d | "
                          "FG %.4f | CG %.4f" %
                          (epoch, num_epoch, batch, num_batch_val * batch_size,
                           np.mean(loss_FG_val), np.mean(loss_CG_val)))

                    if batch % 20 == 0:
                        # Tensorboard 저장하기
                        id = num_batch_val * (epoch - 1) + batch

                        # 4 face with mask
                        mask4f = fn_tonumpy_4(fn_denorm(cm4f, mean=0.5, std=0.5))
                        # 4 face inpaint result
                        result4f = fn_tonumpy_4(fn_denorm(st1_output_inp, mean=0.5, std=0.5))

                        inpainted_cube = fn_tonumpy(fn_denorm(inpainted_cube, mean=0.5, std=0.5))
                        cube = fn_tonumpy(fn_denorm(cube, mean=0.5, std=0.5))
                        x_cube_mask_5 = fn_tonumpy(fn_denorm(x_cube_mask_5, mean=0.5, std=0.5))

                        equirec_ori = c2e(cube[0], h=256, w=512, cube_format='list')
                        equirec_ori_mask = c2e(x_cube_mask_5[0], h=256, w=512, cube_format='list')
                        equirec = c2e(inpainted_cube[0], h=256, w=512, cube_format='list')

                        plt.imsave(os.path.join(result_dir_val, 'png', '%07d_4face_mask.png' % (id)),
                                   mask4f[0], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%07d_4face.png' % (id)),
                                   result4f[0], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%07d_pano_mask.png' % (id)),
                                   equirec_ori_mask, cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%07d_pano.png' % (id)),
                                   equirec, cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%07d_pano_ori.png' % (id)),
                                   equirec_ori, cmap=cmap)

                        writer_val.add_image('4face_mask.png', mask4f[0], id, dataformats='HWC')
                        writer_val.add_image('4face.png', result4f[0], id, dataformats='HWC')
                        writer_val.add_image('pano_ori.png', equirec_ori, id, dataformats='HWC')
                        writer_val.add_image('pano_mask.png', equirec_ori_mask, id, dataformats='HWC')
                        writer_val.add_image('pano.png', equirec, id, dataformats='HWC')

                writer_val.add_scalar('loss_FG', np.mean(loss_FG_val), epoch)
                writer_val.add_scalar('loss_CG', np.mean(loss_CG_val), epoch)

            if epoch % 30 == 0 or epoch == num_epoch:
                face_save(ckpt_dir=ckpt_dir, netFaceG=netFaceG, netFaceD=netFaceD,optimFG=optimFG, optimFD=optimFD,epoch=epoch)
                cube_save(ckpt_dir=ckpt_dir,netCubeG=netCubeG, netWholeD=netWholeD,
                    netSliceD=netSliceD,optimCG=optimCG, optimCD=optimCD,epoch=epoch)

        writer_val.close()
        writer_train.close()

def test(args):
    torch.manual_seed(2020)
    ## test 파라메터 설정하기
    mode = args.mode
    data_parallel = args.data_parallel

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    nch = args.nch
    nker = args.nker

    norm = args.norm

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

    print("norm: %s" % norm)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % gpu)

    ## 디렉토리 생성하기
    result_dir_ori = os.path.join(result_dir, 'ori')
    result_dir_gen = os.path.join(result_dir, 'gen')

    if not os.path.exists(result_dir_ori):
        os.makedirs(result_dir_ori)
    if not os.path.exists(result_dir_gen):
        os.makedirs(result_dir_gen)

    ## 네트워크 학습하기
    if mode == "test":
        transform_test = transforms.Compose([Normalize(), ToTensor()])

        dataset_test = PanoramaDataset(in_dir=os.path.join(data_dir, 'test'), transform=transform_test)
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / 1)

    ## 네트워크 생성하기
    if network == "PInet_new":
        netG = Generator(in_channels=4, out_channels=nch, nker=nker, norm=norm)
        netWholeD = WholeDis(in_channels=4 * 6, out_channels=1, nker=nker, norm=norm)
        netSliceD = SliceDis(in_channels=4, out_channels=1, nker=nker, norm=norm)

        if data_parallel:
            netG = DataParallel(netG)
            netWholeD = DataParallel(netWholeD)
            netSliceD = DataParallel(netSliceD)

        netG = netG.to(gpu)
        netWholeD = netWholeD.to(gpu)
        netSliceD = netSliceD.to(gpu)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netWholeD, init_type='normal', init_gain=0.02)
        init_weights(netSliceD, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    fn_l1 = nn.L1Loss().to(gpu)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    d_params = list(netWholeD.parameters()) + list(netSliceD.parameters())
    optimD = torch.optim.Adam(d_params, lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 1, 3, 4, 2)  # 0,2,3,1
    fn_denorm = lambda x, mean, std: (x * std) + mean

    cmap = None
    id = 0
    # TEST MODE
    if mode == "test":
        netG, netWholeD, netSliceD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                                                    netG=netG, netWholeD=netWholeD, netSliceD=netSliceD,
                                                                    optimG=optimG, optimD=optimD)

        with torch.no_grad():
            netG.eval()
            netWholeD.eval()
            netSliceD.eval()

            loss_G_l1_test = []

            for batch, sample in enumerate(loader_test, 1):
                # forward pass
                cube = sample['cube'].to(gpu, dtype=torch.float32)  # B, F, 3, H, W
                cube_mask = sample['cube_mask'].to(gpu, dtype=torch.float32)  # B, F, 1, H, W

                # cube + cube_mask
                # x_cube_mask -> BN, 6, 3, H, W
                for f in range(6):
                    x_cube_mask_temp = cube[:, f, :, :, :] - cube[:, f, :, :, :] * cube_mask[:, f, :, :, :]
                    x_cube_mask_temp = x_cube_mask_temp.view(x_cube_mask_temp.shape[0], 1,
                                                             x_cube_mask_temp.shape[1],
                                                             x_cube_mask_temp.shape[2], x_cube_mask_temp.shape[3])
                    if f == 0:
                        x_cube_mask = x_cube_mask_temp
                    else:
                        x_cube_mask = torch.cat((x_cube_mask, x_cube_mask_temp), dim=1)

                # x_cube_mask + cube_mask
                # input_cube -> BN, 6*4, H, W
                for f in range(6):
                    input_cube_temp = torch.cat((x_cube_mask[:, f, :, :], cube_mask[:, f, :, :]), dim=1)
                    if f == 0:
                        input_cube = input_cube_temp
                    else:
                        input_cube = torch.cat((input_cube, input_cube_temp), dim=1)
                input_cube = input_cube.to(dtype=torch.float32)

                output = netG(input_cube)

                # output shape change BN, 6*C, H, W -> BN, 6, C, H, W
                # output_5 -> BN, 6, C, H, W
                # Notice, output_5 is output variable dim is 5
                output_cp = output
                for f in range(6):
                    output_temp = output_cp[:, 3 * f:3 * (f + 1), :, :]
                    output_temp = torch.reshape(output_temp, (
                        output_temp.shape[0], 1, output_temp.shape[1], output_temp.shape[2], output_temp.shape[3]))
                    if f == 0:
                        output_5 = output_temp
                    else:
                        output_5 = torch.cat((output_5, output_temp), dim=1)

                # inpainted_result = cube_mask * output + (1 - cube_mask) * cube
                # Using poissonblend, make inpainted_cube
                # inpainted_cube -> BN, 6*3, H, W
                # inpainted_cube = cube_mask * output_5 + (1 - cube_mask) * cube
                for a in range(6):
                    inpainted_cube_temp = output_5[:, a, :, :] * cube_mask[:, a, :, :] + (
                            1 - cube_mask[:, a, :, :]) * cube[:, a, :, :]
                    inpainted_cube_temp = inpainted_cube_temp.unsqueeze(1)
                    if a == 0:
                        inpainted_cube = inpainted_cube_temp
                    else:
                        inpainted_cube = torch.cat((inpainted_cube, inpainted_cube_temp), dim=1)
                inpainted_cube = inpainted_cube.to(gpu, dtype=torch.float32)

                inpainted_cube = fn_tonumpy(fn_denorm(inpainted_cube, mean=0.5, std=0.5))
                cube = fn_tonumpy(fn_denorm(cube, mean=0.5, std=0.5))

                equirec_ori = c2e(cube[0], h=256, w=512, cube_format='list')
                equirec = c2e(inpainted_cube[0], h=256, w=512, cube_format='list')

                # 손실함수 계산
                equirec_ori_tf = torch.from_numpy(equirec_ori)
                equirec_tf = torch.from_numpy(equirec)
                loss_G_l1 = fn_l1(equirec_ori_tf, equirec_tf)

                loss_G_l1_test += [loss_G_l1.item()]

                print("TEST: BATCH %04d / %04d | GEN L1 %.4f" %
                      (batch, num_batch_test * 1,
                       np.mean(loss_G_l1_test)))

                plt.imsave(os.path.join(result_dir_gen, '%07d.png' % (id)),
                           equirec, cmap=cmap)
                plt.imsave(os.path.join(result_dir_ori, '%07d.png' % (id)),
                           equirec_ori, cmap=cmap)

                id += 1

def calc_gradient_penalty(netD, real_data, fake_data, gpu):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)

    alpha = alpha.to(gpu)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_().clone()

    disc_interpolates = netD(interpolates)
    grad_outputs = torch.ones(disc_interpolates.size())

    grad_outputs = grad_outputs.to(gpu)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_outputs, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
