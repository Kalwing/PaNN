import csv
import os
import sys
from glob import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchsummary import summary
from tqdm import tqdm

import dataloader
from config import (
    DATA_NAME,
    PARTIAL_IDS,
    LOSS,
    MODEL_NAME,
    BATCH_MUL,
    N_EPOCH_1,
    N_EPOCH_2_ASC,
    N_EPOCH_2_DESC,
    NET,
    NET_PARAM,
    OPTIM,
    RESULTS_FOLDER,
    TEST_CSV_NAME,
    TRAIN_CSV_NAME,
    TRAINING_NAME,
    SAVE_INTERVAL,
    EARLY_STOP,
    DATASET_TYPE,
    METRICS,
    SEED,
    SECOND_STAGE_ITER,
    SAVE_PRED_INTERVAL,
)
from dataloader import get_dataloader
from utils import compute_distribution_dl
from plot import plot_results, plot_pred, show_data, torch_to_img

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on ", device)

BAR_WIDTH = 110  # Width of the output


def format_fields(fields, precision=5):
    """
    Return a list of str representing given fields with a given precision

    Args:
        fields (List of int, torch.Tensor or dict): Start with an int (the
            epoch number) and is followed by a list of metrics at this epoch
        precision (int, optional): Defaults to 5.

    Returns:
        [List of str]
    """
    r = [fields[0]]
    f = "{:." + f"{precision}f" + "}"
    for field in fields[1:]:
        if not field:
            continue
        elif type(field) is torch.Tensor:
            r.append(f.format(field))
        elif type(field) is dict:
            r.extend([f.format(field[name]) for name in field.keys()])
        else:
            r.append(f.format(field))
    return r


def init_csv(path, val=False, test=False, metrics=None):
    """
    Create a blank csv at the given path, with fields describing the future
    scores of the network

    Args:
        path (Path or str): the path of the csv to be created
        val (bool, optional): if the training will include validation
        test (bool, optional): if the csv is for the test score. val and test
            can't be both true.
        metrics (dict, optional): a dictionary contening the metrics used
            for evaluating the model.
    """
    with open(path, "w") as fout:
        writer = csv.writer(fout)
        fields = ["Epoch"]
        if not test:
            fields.append("Mean Training Loss")
        if val or test:
            name_str = f"Mean {'Val' if val else 'Test'} "
            fields.append(name_str + "Loss")
            if metrics:
                fields.extend([name_str + name for name in metrics.keys()])
        writer.writerow(fields)


class stat_holder:
    def __init__(self, tqdm_instance, len_set=100):
        self.value = 0.0
        self.iter = 0
        self.tqm_instance = tqdm_instance
        self.LOSS_UPDATE_FREQ = len_set // 100 + 1

    def update(self, loss, metrics=None):
        self.value += loss
        if metrics:
            metrics = list(metrics)
            # Initializing the metrics values
            try:
                self.metrics_summed
            except AttributeError:
                self.metrics_summed = {name: 0.0 for name, value in metrics}
            for name, value in metrics:
                self.metrics_summed[name] += value
        # print statistics every LOSS_UPDATE_FREQ mini batches
        if self.iter % self.LOSS_UPDATE_FREQ == 0:
            stats = {"loss": f"{self.value/self.LOSS_UPDATE_FREQ:.5f}"}
            if metrics:
                stats["metrics"] = [
                    f"{name}:" + f"{self.metrics_summed[name]/self.LOSS_UPDATE_FREQ:.5f}"
                    for name in self.metrics_summed.keys()
                ]
                self.metrics_summed = {name: 0.0 for name, value in metrics}
            self.tqm_instance.set_postfix(stats)

            self.value = 0.0
        self.iter += 1


def train_init(
    train_loader,
    net,
    criterion,
    optimizer,
    val_loader=None,
    n_epoch=0,
    metrics=None,
    early_stop_patience=None,
    save_every=None,
    save_path="",
):
    epoch = 0
    i_without_improv = 0
    last_best_train_loss = float("inf")
    train_loss = float("inf")
    while epoch < n_epoch and i_without_improv < early_stop_patience:
        train_progress = tqdm(train_loader, ncols=BAR_WIDTH)
        train_progress.set_description(f"Train {epoch}/{n_epoch}")
        statistics_output = stat_holder(train_progress, len_set=len(train_loader))

        losses = []
        for i, full_partial_data in enumerate(train_progress):  # Loop over the dataset
            data, _ = full_partial_data
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            statistics_output.update(loss)
        train_loss = sum(losses) / i

        # Early Stopping
        if early_stop_patience is not None:
            if not train_loss < last_best_train_loss:
                i_without_improv += 1
            else:
                last_best_train_loss = train_loss
                i_without_improv = 0

        if (
            save_every is not None
            and epoch % save_every == 0
            and i_without_improv < save_every
        ):
            torch.save(
                net.state_dict(),
                RESULTS_FOLDER / TRAINING_NAME / save_path / f"{epoch}_{MODEL_NAME}",
            )
        if val_loader is not None:
            val_loss, val_metrics = run_without_train(val_loader, net, criterion, metrics)
            r = (epoch, train_loss, val_loss, val_metrics)
        else:
            r = (epoch, train_loss)
        if i_without_improv < early_stop_patience:
            yield r
        else:
            return r

        epoch += 1


def train_with_partial(
    train_loader,
    predicted_partials,
    net,
    criterion,
    optimizer,
    val_loader=None,
    n_epoch=0,
    metrics=None,
    early_stop_patience=None,
    save_every=None,
    save_path="",
    ref_distrib=None,
):  # Yeah ok it's ugly and i could refactor train with partial into train_init but TODO I guess
    epoch = 0
    i_without_improv = 0
    last_best_train_loss = float("inf")
    train_loss = float("inf")
    while epoch < n_epoch and i_without_improv < early_stop_patience:
        train_progress = tqdm(train_loader, ncols=BAR_WIDTH)
        train_progress.set_description(f"Train {epoch}/{n_epoch}")
        statistics_output = stat_holder(train_progress, len_set=len(train_loader))
        predicted_loader = iter(predicted_partials)
        losses = []
        for i, full_partial_data in enumerate(train_progress):  # Loop over the dataset
            full_data, partials_data = full_partial_data
            # get the inputs; data is a list of [inputs, labels]
            inputs_full, labels_full = full_data[0].to(device), full_data[1].to(device)
            inputs_partials, labels_partials = list(
                zip(*[(img.to(device), gt.to(device)) for img, gt in partials_data])
            )
            # We suppose that a datalaoder isn't re-shuffled between the moment
            # we predicted the label and the moment we load it w/ the partial
            # inputs
            _, predicted_data = next(predicted_loader)
            inputs_predicted, labels_predicted = list(
                zip(*[(img.to(device), gt.to(device)) for img, gt in predicted_data])
            )

            assert (inputs_predicted[0][0] == inputs_partials[0][0]).all()  # Same image
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs_full = net(inputs_full)
            outputs_partials = [net(inputs_partial) for inputs_partial in inputs_partials]
            loss = criterion(
                outputs_full,
                labels_full,
                outputs_partials,
                labels_partials,
                labels_predicted,
                ref_distrib,
            )
            losses.append(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            statistics_output.update(loss)
        train_loss = sum(losses) / i

        # Early Stopping
        if early_stop_patience is not None:
            if not train_loss < last_best_train_loss:
                i_without_improv += 1
            else:
                last_best_train_loss = train_loss
                i_without_improv = 0

        if (
            save_every is not None
            and epoch % save_every == 0
            and i_without_improv < save_every
        ):
            torch.save(
                net.state_dict(),
                RESULTS_FOLDER / TRAINING_NAME / save_path / f"{epoch}_{MODEL_NAME}",
            )
        if val_loader is not None:
            val_loss, val_metrics = run_without_train(val_loader, net, criterion, metrics)
            r = (epoch, train_loss, val_loss, val_metrics)
        else:
            r = (epoch, train_loss)
        if i_without_improv < early_stop_patience:
            yield r
        else:
            return r

        epoch += 1


def run_without_train(loader, net, criterion, metrics, val=True):
    progress = tqdm(loader, ncols=BAR_WIDTH)
    progress.set_description("Val" if val else "Test")
    statistics_output = stat_holder(progress, len_set=len(loader))

    losses = []

    metrics_names = metrics.keys()
    metrics_values = {name: [] for name in metrics_names}
    with torch.no_grad():
        for i, data in enumerate(progress):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss)

            metrics_val = []
            for name in metrics_names:
                v = metrics[name](outputs.type(torch.float), labels)
                metrics_values[name].append(v)
                metrics_val.append(v)
            # print statistics
            statistics_output.update(loss, zip(metrics_names, metrics_val))
        mean_loss = sum(losses) / len(losses)
        mean_metrics = {
            name: sum(metrics_values[name]) / len(losses) for name in metrics_names
        }
    return mean_loss, mean_metrics


def save_field_in_csv(path, training_data):
    for fields in training_data:
        #  Save values of training in csv
        with open(path, "a") as fout:
            writer = csv.writer(fout)
            writer.writerow(format_fields(fields))


def run(
    train_loader,
    net,
    criterion,
    optimizer,
    val_loader=None,
    metrics=None,
    early_stop_patience=None,
    save_every=None,
):
    # if (RESULTS_FOLDER/TRAINING_NAME/MODEL_NAME).exists():
    #     print("This scenario was already trained", file=sys.stderr)
    #     return

    if not glob(str(RESULTS_FOLDER / TRAINING_NAME / "init" / "*.pth")):
        os.makedirs(RESULTS_FOLDER / TRAINING_NAME / "init", exist_ok=True)
        # train the initial segmentation model
        training = train_init(
            train_loader,
            net,
            criterion,
            optimizer,
            val_loader,
            N_EPOCH_1,
            metrics,
            early_stop_patience=early_stop_patience,
            save_every=save_every,
            save_path="init",
        )
        #  Create the appropriate fields in the csv
        init_csv(
            RESULTS_FOLDER / TRAINING_NAME / "init" / TRAIN_CSV_NAME,
            val=True,
            metrics=metrics,
        )
        save_field_in_csv(
            RESULTS_FOLDER / TRAINING_NAME / "init" / TRAIN_CSV_NAME, training
        )  # The training is done here
        torch.save(
            net.state_dict(), RESULTS_FOLDER / TRAINING_NAME / "init" / MODEL_NAME,
        )

        plot_pred(
            val_loader.dataset,
            net,
            n=3,
            device=device,
            save_path=RESULTS_FOLDER / TRAINING_NAME / "init",
        )
        print("Finished init training")  # TODO: log final result of training

    else:
        init_path = RESULTS_FOLDER / TRAINING_NAME / "init"
        models = [path for path in init_path.iterdir() if path.suffix == ".pth"]
        initfile = sorted(models)[-1]
        print("Loaded:", initfile)
        net.load_state_dict(torch.load(initfile))

    # Compute the prior distribution
    print("-> Computing the initial parameters associated to the primal variables:")
    q = compute_distribution_dl(train_loader.full_loader).to(device)
    # TODO: output q
    net.nu.copy_(-1 / (q + 1e-10))
    net.mu.copy_(-1 / (1 - q + 1e-10))

    # Update segmentation model
    criterion.use_partial()
    print("SECOND STAGE")
    for iter in range(SECOND_STAGE_ITER):
        base_path = RESULTS_FOLDER / TRAINING_NAME / "2nd_stage" / str(iter)

        # Compute YP
        if iter % SAVE_PRED_INTERVAL == 0:
            with torch.no_grad():
                pred_path = RESULTS_FOLDER / TRAINING_NAME / "Predicted"
                dataloader.save_preds(
                    train_loader, net, pred_path, device=device,
                )
                predicted_partials = [
                    dataloader.ImgPredDataset(
                        str(partial),
                        base_path=pred_path,
                        split_type="train",
                        randomize=False,
                    )
                    for partial in PARTIAL_IDS
                ]
                pred_label_loader = dataloader.ZhouLoader(
                    train_loader.full_loader.dataset,
                    predicted_partials,
                    batch_mul=BATCH_MUL,
                )

        # update mu and nu
        print(f"ASCENT {iter+1}/{SECOND_STAGE_ITER}")
        net.switch_to_loss_training(True)
        criterion.switch_to_ascent()

        training = train_with_partial(
            train_loader,
            pred_label_loader,
            net,
            criterion,
            optimizer,
            val_loader,
            N_EPOCH_2_ASC,
            metrics,
            early_stop_patience=early_stop_patience,
            save_every=save_every,
            save_path="init",
            ref_distrib=q,
        )
        os.makedirs(base_path / "ascent", exist_ok=True)
        init_csv(
            base_path / "ascent" / TRAIN_CSV_NAME, val=True, metrics=metrics,
        )
        save_field_in_csv(base_path / "ascent" / TRAIN_CSV_NAME, training)

        # Update weights
        print(f"DESCENT {iter+1}/{SECOND_STAGE_ITER}")
        net.switch_to_loss_training(False)
        criterion.use_partial()

        training = train_with_partial(
            train_loader,
            pred_label_loader,
            net,
            criterion,
            optimizer,
            val_loader,
            N_EPOCH_2_DESC,
            metrics,
            early_stop_patience=early_stop_patience,
            save_every=save_every,
            save_path="init",
            ref_distrib=q,
        )
        os.makedirs(base_path / "descent", exist_ok=True)
        init_csv(
            base_path / "descent" / TRAIN_CSV_NAME, val=True, metrics=metrics,
        )
        save_field_in_csv(base_path / "descent" / TRAIN_CSV_NAME, training)

        plot_pred(
            val_loader.dataset, net, n=5, device=device, save_path=base_path,
        )
    torch.save(net.state_dict(), RESULTS_FOLDER / TRAINING_NAME / MODEL_NAME)
    return [RESULTS_FOLDER / TRAINING_NAME / "init" / TRAIN_CSV_NAME]


if __name__ == "__main__":
    # DATALOADERS
    exec(f"full_train_set = dataloader.{DATASET_TYPE}('Full', split_type='train')")
    exec(
        f"partial_train_set = [ \
            dataloader.{DATASET_TYPE}(str(partial), split_type='train') for partial in PARTIAL_IDS \
        ]"
    )
    train_loader = dataloader.ZhouLoader(
        full_train_set, partial_train_set, batch_mul=BATCH_MUL
    )  # TODO: compute batch_mul based on batch size

    exec(f"val_set = dataloader.{DATASET_TYPE}('Full', split_type='val')")
    val_loader = get_dataloader(val_set)
    exec(f"test_set = dataloader.{DATASET_TYPE}('Full', split_type='test')")
    test_loader = get_dataloader(test_set)

    # show_data(full_train_set, partial_train_set)
    # NEURAL NET SETUP
    net = NET(**NET_PARAM)
    net.to(device)
    criterion = LOSS(net)
    optimizer = OPTIM(net.parameters())  # TODO: Different optimizer for each stage
    # Outputing the parameters of the networks in the results folder
    try:
        os.makedirs(RESULTS_FOLDER / TRAINING_NAME)
    except FileExistsError:
        pass
    with open(RESULTS_FOLDER / TRAINING_NAME / "summary.md", "w") as fout:
        shape = full_train_set[0][0].shape
        shape = (1, 128, 128)
        fout.write(f"Trained the {datetime.now()}\n\n")
        fout.write(f"\n*Data*: `{DATA_NAME}`  \n")
        fout.write(f"\n*Network*: \n```python\n{net}\n```  ")
        fout.write(f"\n*Loss*: `{criterion}`  ")
        fout.write(f"\n*Optimizer*: \n```python\n{optimizer}\n```  ")
        s = sys.stdout
        sys.stdout = fout  #  A bit hacky, but allow summary to be printed in fout
        fout.write(f"\n\n*Summary*: (for input of size {shape})\n```js\n")
        summary(net, shape)
        sys.stdout = s
        fout.write(f"\n```")

    metrics = METRICS
    # TRAINING LOOP
    csv_files = run(
        train_loader,
        net,
        criterion,
        optimizer,
        val_loader=val_loader,
        metrics=metrics,
        early_stop_patience=EARLY_STOP,
        save_every=SAVE_INTERVAL,
    )
    plot_results(csv_files, save_path=RESULTS_FOLDER / TRAINING_NAME)

    plot_pred(
        test_set, net, n=3, device=device, save_path=RESULTS_FOLDER / TRAINING_NAME,
    )

    #  TEST EVALUATION
    test_results = run_without_train(
        test_loader, net, criterion, val=False, metrics=metrics
    )

    init_csv(
        RESULTS_FOLDER / TRAINING_NAME / TEST_CSV_NAME, test=True, metrics=metrics,
    )
    with open(RESULTS_FOLDER / TRAINING_NAME / TEST_CSV_NAME, "a") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            format_fields([N_EPOCH_1 + N_EPOCH_2_ASC + N_EPOCH_2_DESC, *test_results])
        )
