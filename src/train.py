import os
import re
from random import randint
import uuid
from itertools import product

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler, sample_transformation
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True

def train_step(model, xs, ys, optimizer, loss_func, prompt_type, prompt_row, prompt_col):
    optimizer.zero_grad()
    _, _, output = model(xs, ys, prompt_type, prompt_row, prompt_col)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def create_schedule(num_tasks, train_steps):
    steps_per_task = train_steps // num_tasks
    remaining_steps = train_steps % num_tasks
    task_intervals = [steps_per_task] * num_tasks

    for i in range(remaining_steps):
        task_intervals[i] += 1

    task_schedule = [sum(task_intervals[:i + 1]) for i in range(num_tasks)]

    return task_schedule

def select_task(current_step, num_tasks, task_schedule, task_breakdown):
    task_id = next((i for i, entry in enumerate(task_breakdown) if entry >= current_step), None)
    if task_schedule == 'random':
        return randint(0, num_tasks - 1)
    elif task_schedule == 'sequential':
        return task_id
    elif task_schedule == 'mixed_sequential':
        return randint(0, task_id)

def select_data(current_step, num_dist, data_split, data_schedule, task_schedule, task_breakdown, data_breakdown):
    data_id = None
    if data_split == 'blocks_based' and task_schedule != 'random':
        block_index = next((i for i, entry in enumerate(task_breakdown) if entry >= current_step), None)
        block_max = task_breakdown[block_index]
        block_breakdown = create_schedule(num_dist, block_max)
        data_id = next((i for i, entry in enumerate(block_breakdown) if entry >= current_step), None)
    elif data_split == 'steps_based':
        data_id = next((i for i, entry in enumerate(data_breakdown) if entry >= current_step), None)

    if data_schedule == 'random':
        return randint(0, num_dist - 1)
    elif data_schedule == 'sequential':
        return data_id
    elif data_schedule == 'mixed_sequential':
        return randint(0, data_id)

def run_validation(model, n_points, bsize, current_step, loss_func, prompt_type, prompt_args,
                   data_list, task_list):
    loss_dict = {f"{key1}_{key2}": list() for key1, key2 in product(task_list.keys(), data_list.keys())}

    for i in range(1, 501):
        start_seed = i * -64
        set_seed = [j for j in range(start_seed, start_seed + 64)]

        data_sampler_args = {'seeds': set_seed}
        task_sampler_args = {'seeds': set_seed}

        prompt_col = 0

        for data_key in data_list:
            data_sampler = data_list[data_key]
            data_pos = int(re.search(r'\d+', data_key).group(0))

            xs = data_sampler.sample_xs(
                n_points,
                bsize,
                **data_sampler_args,
            )

            for key in task_list:
                task = task_list[key](**task_sampler_args)
                task_pos = int(re.search(r'\d+', key).group(0))

                ys = task.evaluate(xs)

                prompt_row = 0
                if len(prompt_args) != 0:
                    if prompt_args['type'] == 'data':
                        if prompt_args['encoding'] == 'dynamic':
                            prompt_row = data_pos
                        prompt_col = prompt_args['position']
                    elif prompt_args['type'] == 'task':
                        if prompt_args['encoding'] == 'dynamic':
                            prompt_row = task_pos
                        prompt_col = prompt_args['position']

                with torch.no_grad():
                    _, _, prediction = model(xs.cpu(), ys.cpu(),
                                       prompt_type, prompt_row, prompt_col)
                loss = loss_func(prediction, ys.cpu())

                loss_dict[f'task_{task_pos}_data_{data_pos}'].append(loss.mean(axis=0))

    for key in loss_dict:
        wandb.log({
            f'{key} 10-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[9],
            f'{key} 20-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[19],
            f'{key} 30-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[29],
            f'{key} 40-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[39],
            f'{key} 50-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[49],
            f'{key} 60-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[59],
            f'{key} 70-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[69],
            f'{key} 80-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[79],
            f'{key} 90-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[89],
            f'{key} 100-shot loss': (sum(loss_dict[key]) / len(loss_dict[key]))[99]
        }, step=current_step)

def train(model, args, file_mse):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]

    n_dims = model.n_dims
    bsize = args.training.batch_size
    n_points = args.training.n_points

    task_list = {}
    num_tasks = 0
    task_schedule = args.training.task_schedule

    data_list = {}
    num_dist = 0
    data_split = args.training.data_split
    data_schedule = args.training.data_schedule

    for key in args.training.task_kwargs:
        num_tasks += 1
        task_list[key] = get_task_sampler(
            args.training.task_kwargs[key]['task'],
            n_dims,
            bsize,
            weight_type=args.training.weight_type,
            **{i: args.training.task_kwargs[key][i] for i in args.training.task_kwargs[key] if i != 'task'},
        )
    task_breakdown = create_schedule(num_tasks, args.training.train_steps)

    for key in args.training.data_kwargs:
        single_sample = args.training.data_kwargs[key]
        num_dist += 1
        sample_args = {}

        if single_sample['type'] == 'skewed':
            eigenvals = 1 / (torch.arange(n_dims) + 1)
            scale = sample_transformation(eigenvals, normalize=True)
            sample_args['scale'] = scale

        for info in single_sample:
            if info != 'data' and info != 'type':
                sample_args[info] = single_sample[info]

        data_list[key] = get_data_sampler(single_sample['data'], n_dims=n_dims, **sample_args)
    data_breakdown = create_schedule(num_dist, args.training.train_steps)

    pbar = tqdm(range(starting_step, args.training.train_steps))

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        sample_id = select_data(current_step=i, num_dist=num_dist, data_split=data_split,
                                data_schedule=data_schedule, task_schedule=task_schedule,
                                task_breakdown=task_breakdown, data_breakdown=data_breakdown)

        data_sampler = data_list[f"data_{sample_id}"]

        xs = data_sampler.sample_xs(
            n_points,
            bsize,
            **data_sampler_args,
        )

        task_id = select_task(current_step=i, num_tasks=num_tasks, task_schedule=task_schedule,
                              task_breakdown=task_breakdown)

        task = task_list[f"task_{task_id}"](**task_sampler_args)

        ys = task.evaluate(xs)
        loss_func = task.get_training_metric()

        prompt_row = 0
        prompt_col = 0
        if len(args.training.prompt_kwargs) != 0:
            if args.training.prompt_kwargs['type'] == 'data':
                if args.training.prompt_kwargs['encoding'] == 'dynamic':
                    prompt_row = sample_id
                prompt_col = args.training.prompt_kwargs['position']
            elif args.training.prompt_kwargs['type'] == 'task':
                if args.training.prompt_kwargs['encoding'] == 'dynamic':
                    prompt_row = task_id
                prompt_col = args.training.prompt_kwargs['position']

        loss, output = train_step(model, xs.cpu(), ys.cpu(), optimizer, loss_func,
                                  args.training.prompt_type, prompt_row, prompt_col)
        file_mse.write(f"{task_id}\t{loss}\n")

        point_wise_tags = list(range(n_points))
        point_wise_loss_func = task.get_metric()

        point_wise_loss = point_wise_loss_func(output, ys.cpu()).mean(dim=0)

        if i % 2000 == 0:
           val_type = args.training.val_type
           if val_type == 'current_tasks':
               run_validation(model=model, n_points=n_points, bsize=bsize, current_step=i,
                              loss_func=point_wise_loss_func, prompt_type=args.training.prompt_type,
                              prompt_args=args.training.prompt_kwargs, data_list=data_list,task_list=task_list)
           elif val_type == 'provided_tasks':
               val_list = {}
               val_kwargs = args.training.validation_kwargs
               for key in args.training.validation_kwargs:
                   val_list[key] = get_task_sampler(
                       val_kwargs[key]['task'],
                       n_dims,
                       bsize,
                       weight_type=args.training.weight_type,
                       num_tasks=args.training.num_tasks,
                       **{i: val_kwargs[key][i] for i in val_kwargs[key] if i != 'task'},
                   )
               run_validation(model=model, n_points=n_points, bsize=bsize, current_step=i,
                              loss_func=point_wise_loss_func, prompt_type=args.training.prompt_type,
                              prompt_args=args.training.prompt_kwargs, data_list=data_list,task_list=val_list)

        baseline_loss = (
                sum(
                    max(n_points - ii, 0)
                    for ii in range(n_points)
                )
                / n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": n_points,
                    "n_dims": n_dims,
                },
                step=i,
            )

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
                args.training.keep_every_steps > 0
                and i % args.training.keep_every_steps == 0
                and not args.test_run
                and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)

    model.cpu()
    model.train()

    file_mse = open(f"../models/seed_{args.model.transformer_seed}_{args.training.task_schedule}_mse.txt", "w")
    train(model, args, file_mse)
    file_mse.close()

    # if not args.test_run:
    #    _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    # Enter api key for wandb under variable api_key
    wandb.login(key=api_key)

    main(args)
