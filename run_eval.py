from pathlib import Path
import subprocess
import argparse

ckpts = [
    # "iiiqnrs0/mocov3nf_lr16-iiiqnrs0-ep=last.ckpt",
    # "q9y4x1nm/mocov3nf_lr16_c£enter-q9y4x1nm-ep=last.ckpt",
    # "mocov3nf_lr16_gs540resize-mv0pnuza-ep=last.ckpt",
    # "mocov3nf_lr16t0-yboaa3by-ep=last.ckpt",
    # "mocov3nf_lr16t20-zd0jdcjy-ep=last.ckpt",
    # "mocov3nf_lr16t5-1qc1r931-ep=last.ckpt",
    # "mocov3nf_lr16t10-vpi2u0mn-ep=last.ckpt",
    # "ba1e2na0/mocov3nf_lr16_gs540-ba1e2na0-ep=last.ckpt",
    # "mocov3nf_lr16_gs448-i477bc9q-ep=last.ckpt",
    # "mocov3nf_lr16_gs336-w5mbrmup-ep=last.ckpt",
    # "mocov3nf_lr16_gs112-rsl7me4h-ep=last.ckpt",
    # "mocov3nf_lr16t25-qo9qs3j3-ep=last.ckpt",
    # "tlyydhj5/mocov3nf_lr16_gs336_t5-tlyydhj5-ep=last.ckpt"
    # 'm4lf6jxv/mocov3nf_lr16_gs540_resize_t5-m4lf6jxv-ep=last.ckpt'.
    # "ewnydxom/mocov3nf_lr16_gs540_t0-ewnydxom-ep=last.ckpt",
    # "3aah1qh5/mocov3nf_lr16_gs313x2-3aah1qh5-ep=last.ckpt",
    # 'tl8ynbj4/mocov3nf_lr16_gs540_t15x2-tl8ynbj4-ep=last-stp=last.ckpt',
    # 'ixgzlby3/mocov3nf_lr16_gs224_t15x2-ixgzlby3-ep=last-stp=last.ckpt',
    'blmypuhb/mocov3nf_lr16_gsRND_t15-blmypuhb-ep=last-stp=last.ckpt'
]

datasets = [
    'imagenet_42',
    'imagenet100_42',
    'imagenet100_im',
    'DTD',
    'Flowers102',
    'FGVCAircraft',
    'OxfordIIITPet',
    'StanfordCars',
    'cifar10_224',
    'cifar100_224',
    'toybox',
    'core50',
    'COIL100',
    'imagenet10pct_42',
    'imagenet1pct_42',
    'Places365_h5',
    'STL10_224',
    # 'STL10_FG_224',
    # 'STL10',
    # 'STL10_FG',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="42")
    args = parser.parse_args()

    env = args.env
    print(f"Running on {env} environment")
    if env == "42":
        scratch_root = Path("/pfss/mlde/workspaces/mlde_wsp_PI_Roig/schaumloeffel/logs/ego4d/mocov3/")
    else:
        scratch_root = Path("/scratch/modelrep/schaumloeffel/ego4d_backup")
        # ckpts = [Path(ckpt).name for ckpt in ckpts]  # remove the folder name

    lrs = [1.0]
    devices = 8
    bs = 64
    optimizer = "lars"

    script = "mocov3_ts.yaml" if env == "42" else "mocov3_ts_hlr.yaml"
    # base_cmd = "python main_linear.py --config-path scripts/linear_probe/ --config-name moco_core50.yaml"
    base_cmd = f"python main_linear.py --config-path scripts/linear_probe/ --config-name {script}"
    args = "++name={name} ++pretrained_feature_extractor=\"{ft}\" ++data.dataset={ds} ++optimizer.lr={lr} ++wandb.job_type={job_type} ++devices={devices} ++optimizer.batch_size={bs}"
    args += " ++early_stopping.enabled=False ++optimizer.name={optimizer}"

    # job_type = "linear_probe" if len(lrs) == 1 else "find_lr"
    job_type = "bg_exps"

    for dataset in datasets:
        for ckpt in ckpts:
            for lr in lrs:
                if dataset == "core50":
                    lr = 0.3
                if dataset == "imagenet100_42":
                    lr = 0.3
                if dataset == "imagenet_42":
                    lr = 0.3
                if dataset == "COIL100":
                    devices = 8
                    bs = 12
                    optimizer = "sgd"
                name_prefix = dataset + '_'
                name_suffix = '_lr' + str(lr)
                name = Path(ckpt).stem.split('-')[0]
                full_ckpt = str(scratch_root / ckpt).replace("=", "\=")
                cmd = base_cmd + " " + args.format(name=name_prefix + name + name_suffix, ft=full_ckpt, ds=dataset,
                                                   lr=lr,
                                                   job_type=job_type, devices=devices, bs=bs, optimizer=optimizer)
                subprocess.run(cmd, shell=True)
