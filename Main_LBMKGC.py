import torch
from mmkgc.config import Tester, LBTrainer
from mmkgc.module.model import AdvRelRotatE
from mmkgc.module.loss import SigmoidLoss
from mmkgc.module.strategy import NegativeSampling
from mmkgc.data import TrainDataLoader, TestDataLoader
from args import get_args

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    # # set the seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/" + args.dataset + '/',
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )
    
    # dataloader for test
    test_dataloader = TestDataLoader(
        "./benchmarks/" + args.dataset + '/', "link")
    img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')
    
    # define the model
    kge_score = AdvRelRotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        margin=args.margin,
        epsilon=2.0,
        img_emb=img_emb,
        text_emb=text_emb
    )
    
    # define the loss function
    model = NegativeSampling(
        model=kge_score,
        loss=SigmoidLoss(adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=0.00001
    ) 
    
    # train the model
    trainer = LBTrainer(
        model=model,
        data_loader=train_dataloader,
        test_data_loader=test_dataloader,
        train_times=args.epoch,
        alpha=args.learning_rate,
        use_gpu=True,
        opt_method='Adam'
    )
    
    trainer.run()
    
    # kge_score.save_checkpoint('./checkpoint/' + args.save)
    
    # test the model
    # kge_score.load_checkpoint('./checkpoint/' + args.save)
    
    print("=============================================Test=============================================")
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)
