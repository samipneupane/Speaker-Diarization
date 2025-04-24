# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini)
# Licensed under the MIT license.


from eend.backend.models import (
    average_checkpoints,
    get_model,
    load_checkpoint,
    pad_labels,
    pad_sequence,
    save_checkpoint,
)
from eend.backend.updater import setup_optimizer, get_rate
from eend.common_utils.diarization_dataset import KaldiDiarizationDataset
from eend.common_utils.gpu_utils import use_single_gpu
from eend.common_utils.metrics import (
    calculate_metrics,
    new_metrics,
    reset_metrics,
    update_metrics,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import os
import random
import torch
import logging
import yamlargparse
import csv


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _convert(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Any]:
    return {'xs': [x for x, _, _ in batch],
            'ts': [t for _, t, _ in batch],
            'names': [r for _, _, r in batch]}


def compute_loss_and_metrics(
    model: torch.nn.Module,
    labels: torch.Tensor,
    input: torch.Tensor,
    n_speakers: List[int],
    acum_metrics: Dict[str, float],
    vad_loss_weight: float,
    detach_attractor_loss: bool
) -> Tuple[torch.Tensor, Dict[str, float]]:
    y_pred, attractor_loss = model(input, labels, n_speakers, args)
    loss, standard_loss = model.get_loss(
        y_pred, labels, n_speakers, attractor_loss, vad_loss_weight,
        detach_attractor_loss)
    # print("sl - ", standard_loss) #####
    # print("al - ", attractor_loss) #####
    metrics = calculate_metrics(
        labels.detach(), y_pred.detach(), threshold=0.5)
    #print(metrics) #####
    acum_metrics = update_metrics(acum_metrics, metrics)
    #print("loss = ", loss) ###
    #print("a_m = ", acum_metrics['loss']) ###
    acum_metrics['loss'] += loss.item()
    #print("a_m = ", acum_metrics['loss']) ###
    acum_metrics['loss_standard'] += standard_loss.item()
    acum_metrics['loss_attractor'] += attractor_loss.item()
    return loss, acum_metrics


def get_training_dataloaders(
    args: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    train_set = KaldiDiarizationDataset(
        args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batchsize,
        collate_fn=_convert,
        num_workers=args.num_workers,
        shuffle=True,
        worker_init_fn=_init_fn,
    )

    dev_set = KaldiDiarizationDataset(
        args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.dev_batchsize,
        collate_fn=_convert,
        num_workers=1,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    Y_train, _, _ = train_set.__getitem__(0)
    Y_dev, _, _ = dev_set.__getitem__(0)
    assert Y_train.shape[1] == Y_dev.shape[1], \
        f"Train features dimensionality ({Y_train.shape[1]}) and \
        dev features dimensionality ({Y_dev.shape[1]}) differ."
    assert Y_train.shape[1] == (
        args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of {args.feature_dim} \
        but {Y_train.shape[1]} found."

    return train_loader, dev_loader


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND training')
    #parser.add_argument('-c', '--config', help='config file path', action=yamlargparse.ActionConfigFile) #####
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--dev-batchsize', default=1, type=int,
                        help='number of utterances in one development batch')
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', default=-1, type=int,
                        help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--init-epochs', type=str, default='',
                        help='Initialize model with average of epochs \
                        separated by commas or - for intervals.')
    parser.add_argument('--init-model-path', type=str, default='',
                        help='Initialize the model from the given directory')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max-epochs', type=int,
                        help='Max. number of epochs to train')
    parser.add_argument('--min-length', default=0, type=int,
                        help='Minimum number of frames for the sequences'
                             ' after downsampling.')
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--noam-warmup-steps', default=100000, type=float)
    parser.add_argument('--num-frames', default=500, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int,
                        help='maximum number of speakers allowed')
    parser.add_argument('--num-workers', default=1, type=int,
                        help='number of workers in train DataLoader')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--train-batchsize', default=1, type=int,
                        help='number of utterances in one train batch')
    parser.add_argument('--train-data-dir', type = str,
                        help='kaldi-style data dir used for training.')
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--use-last-samples', default=True, type=bool)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    parser.add_argument('--valid-data-dir',
                        help='kaldi-style data dir used for validation.')

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument(
        '--attractor-loss-ratio', default=1.0, type=float,
        help='weighting parameter')
    attractor_args.add_argument(
        '--attractor-encoder-dropout', type=float)
    attractor_args.add_argument(
        '--attractor-decoder-dropout', type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', type=bool,
        help='If True, avoid backpropagation on attractor loss')

    args = parser.parse_args()

    #####
    import yaml
    config_path = 'yaml/adapt.yaml' #####
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    #####

    args.init_model_path = "model/github/LibriSpeech"
    args.init_epochs = "90-100"
    args.max_epochs = 1
    args.num_speakers = 4
    args.output_path = "model/trained_model/checkspk"
    args.train_data_dir = "dataset/merged_audio/train/details"
    args.valid_data_dir = "dataset/merged_audio/validation/details"

    return args


def save_metrics_to_csv(filename, metrics, epoch, batch_qty):
    csv_columns = ['epoch'] + list(metrics.keys())
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        if file.tell() == 0:
            writer.writeheader()
        metrics_with_epoch = {'epoch': epoch}
        for k, v in metrics.items():
            metrics_with_epoch[k] = v / batch_qty
        writer.writerow(metrics_with_epoch)



if __name__ == '__main__':
    args = parse_arguments()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    logging.info(args)

    writer = SummaryWriter(f"{args.output_path}/tensorboard")

    train_loader, dev_loader = get_training_dataloaders(args)

    if args.gpu >= 1:
        gpuid = use_single_gpu(args.gpu)
        logging.info('GPU device {} is used'.format(gpuid))
        args.device = torch.device("cuda")
    else:
        gpuid = -1
        args.device = torch.device("cpu")

    if args.init_model_path == '':
        model = get_model(args)
        optimizer = setup_optimizer(args, model)
    else:
        model = get_model(args)
        model = average_checkpoints(
            args.device, model, args.init_model_path, args.init_epochs)
        optimizer = setup_optimizer(args, model)

    train_batches_qty = len(train_loader)
    dev_batches_qty = len(dev_loader)

    logging.info(f"#batches quantity for train: {train_batches_qty}")
    logging.info(f"#batches quantity for dev: {dev_batches_qty}")

    # Initialize metrics
    acum_train_metrics = new_metrics()
    acum_dev_metrics = new_metrics()

    if os.path.isfile(os.path.join(
            args.output_path, 'models', 'checkpoint_0.tar')):
        # Load latest model and continue from there
        directory = os.path.join(args.output_path, 'models')
        checkpoints = os.listdir(directory)
        paths = [os.path.join(directory, basename) for
                 basename in checkpoints if basename.startswith("checkpoint_")]
        latest = max(paths, key=os.path.getctime)
        epoch, model, optimizer, _ = load_checkpoint(args, latest)
        init_epoch = epoch
    else:
        init_epoch = 0
        # Save initial model
        save_checkpoint(args, init_epoch, model, optimizer, 0)

    csv_columns = ['epoch', 'loss', 'loss_standard', 'loss_attractor', 'avg_ref_spk_qty', 'avg_pred_spk_qty', 
                'DER_FA', 'DER_miss', 'VAD_FA', 'VAD_miss', 'OSD_FA', 'OSD_miss']

    for epoch in range(init_epoch, args.max_epochs):
        #####
        from datetime import datetime
        current_time = datetime.now()
        print("Current Device Time:", current_time)
        print(f"Epoch {epoch}/{args.max_epochs - 1}")
        #####
        
        model.train()

        for i, batch in enumerate(train_loader):
            features = batch['xs']
            labels = batch['ts']

            ################
            '''
            print(features[0])
            print(labels[1])

            print(features[2].shape)
            print(labels[2].shape)
            print("Hi", len(features))
            print("Hi", len(labels)) 

            labels[0] = torch.tensor([[0, 1, 0, 0, 0]] * 128, dtype=torch.int32)
            labels[0][100] = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
            labels[0][101] = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
            labels[0][102] = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
            labels[0][103] = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)

            labels[2] = torch.tensor([[0, 0, 1, 0, 0]] * 128, dtype=torch.int32)
            '''
            #############
            # print("labels = ", labels) #####
            
            n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                     if t.sum() > 0 else 0 for t in labels])
            # print("n_speakers = ", n_speakers) #####
            max_n_speakers = max(n_speakers)
            # print("max_n_speakers = ", max_n_speakers) #####
            features, labels = pad_sequence(features, labels, args.num_frames)
            labels = pad_labels(labels, max_n_speakers)
            features = torch.stack(features).to(args.device)
            labels = torch.stack(labels).to(args.device)

            loss, acum_train_metrics = compute_loss_and_metrics(
                model, labels, features, n_speakers, acum_train_metrics,
                args.vad_loss_weight,
                args.detach_attractor_loss)

            if i % args.log_report_batches_num == \
                    (args.log_report_batches_num-1):
                for k in acum_train_metrics.keys():
                    writer.add_scalar(
                        f"train_{k}",
                        acum_train_metrics[k] / args.log_report_batches_num,
                        epoch * train_batches_qty + i)
                writer.add_scalar(
                    "lrate",
                    get_rate(optimizer),
                    epoch * train_batches_qty + i)
                acum_train_metrics = reset_metrics(acum_train_metrics)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
            optimizer.step()

        # Save checkpoint
        save_checkpoint(args, epoch+1, model, optimizer, loss)

        # Print training metrics
        print(f"Training metrics for epoch {epoch}:")
        for k, v in acum_train_metrics.items():
            print(f"  {k}: {v / train_batches_qty:.4f}") #/32
        save_metrics_to_csv("model/trained_model/checkspk/training_metrics.csv", acum_train_metrics, epoch, train_batches_qty)


        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(dev_loader):
                features = batch['xs']
                labels = batch['ts']
                n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                        if t.sum() > 0 else 0 for t in labels])
                max_n_speakers = max(n_speakers)
                features, labels = pad_sequence(
                    features, labels, args.num_frames)
                labels = pad_labels(labels, max_n_speakers)
                features = torch.stack(features).to(args.device)
                labels = torch.stack(labels).to(args.device)
                _, acum_dev_metrics = compute_loss_and_metrics(
                    model, labels, features, n_speakers, acum_dev_metrics,
                    args.vad_loss_weight,
                    args.detach_attractor_loss)

        for k in acum_dev_metrics.keys():
            writer.add_scalar(
                f"dev_{k}", acum_dev_metrics[k] / dev_batches_qty,
                epoch * dev_batches_qty + i)
        
        # Print validation metrics
        print(f"Validation metrics for epoch {epoch}:")
        for k, v in acum_dev_metrics.items():
            print(f"  {k}: {v / dev_batches_qty:.4f}")
        save_metrics_to_csv("model/trained_model/checkspk/validation_metrics.csv", acum_dev_metrics, epoch, dev_batches_qty)

        acum_dev_metrics = reset_metrics(acum_dev_metrics)
