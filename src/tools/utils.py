import os
from PIL import Image
import torch
import torch.utils.model_zoo
from torch.utils.data import Dataset
import torchvision
import math
import numpy as np
import random
from sklearn.metrics import precision_recall_curve
import redis
import io
import tqdm
import dataclasses
from torch.profiler import record_function


def least_common_multiple(num):
    mini = 1
    for i in num:
        mini = int(i) * int(mini) / math.gcd(int(i), mini)
        mini = int(mini)
    return mini


def serialize_tensor(tensor):
    """Serialize a tensor to a byte array"""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


@dataclasses.dataclass
class RedisConfig:

    redis_host: str
    redis_port: int
    redis_db_num: int


def preload_dataset_to_redis(
    data_dir, exp_index_list: list, redis_config: RedisConfig, reuse: bool = False
) -> None:

    # get redis connection
    try:
        r = redis.Redis(
            host=redis_config.redis_host,
            port=redis_config.redis_port,
            db=redis_config.redis_db_num,
        )
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        exit(1)

    # check if the db is empty
    if r.dbsize() > 0:
        print(
            "The Redis database is not empty. Recommend clearing the database before preloading the dataset."
        )
        if not reuse:
            r.close()
            raise ValueError("The Redis database is not empty but reuse is not set.")
        else:
            print("Proceeding with existing data...")
            r.close()
            return
    
    pipe = r.pipeline(transaction=False)

    total_aps = [
        np.sort(os.listdir(data_dir + str(idx) + "/dvs_frames"))
        for idx in exp_index_list
    ]
    total_dvs = [
        np.sort(os.listdir(data_dir + str(idx) + "/dvs_7ms_3seq"))
        for idx in exp_index_list
    ]
    num_aps = [len(x) for x in total_aps]
    num_dvs = [len(x) for x in total_dvs]
    num_exp = len(exp_index_list)

    # preload image frames
    print("Preloading DVS frames...")
    for exp_index in range(num_exp):
        for idx in tqdm.tqdm(
            range(num_aps[exp_index]), desc=f"Exp {exp_index_list[exp_index]}"
        ):
            img_loc = (
                data_dir
                + str(exp_index_list[exp_index])
                + "/dvs_frames/"
                + total_aps[exp_index][idx]
            )
            img = Image.open(img_loc).convert("RGB")

            # convert to tensor
            curr_tensor = torchvision.transforms.ToTensor()(img)
            
            # close the image
            img.close()

            # key - 1. "image_tensor" 2. "{exp_idx}_{idx}"
            # use hash to store the tensor
            pipe.hset(
                "image_tensor",
                f"{exp_index_list[exp_index]}_{idx}",
                serialize_tensor(curr_tensor),
            )
    
    pipe.execute()

    # preload dvs tensors
    print("Preloading DVS tensors...")
    for exp_index in range(num_exp):
        for idx in tqdm.tqdm(
            range(num_dvs[exp_index]), desc=f"Exp {exp_index_list[exp_index]}"
        ):
            dvs_path = (
                data_dir
                + str(exp_index_list[exp_index])
                + "/dvs_7ms_3seq/"
                + total_dvs[exp_index][idx]
            )
            dvs_buf = torch.load(dvs_path, weights_only=True, map_location="cpu")

            # preprocess the tensor
            dvs_buf = dvs_buf.permute([1, 0, 2, 3])
            dvs_buf = torch.nn.functional.avg_pool2d(dvs_buf, 2)

            # key - 1. "dvs_tensor" 2. "{exp_idx}_{idx}"
            # use hash to store the tensor
            pipe.hset(
                "dvs_tensor",
                f"{exp_index_list[exp_index]}_{idx}",
                serialize_tensor(dvs_buf),
            )
    
    pipe.execute()

    print("Preloading completed.")

    # disconnect from redis
    pipe.close()
    r.close()

    return


class SequentialDataset(Dataset):
    def __init__(
        self,
        data_dir,
        exp_idx_list,
        transform,
        nclass=None,
        seq_len_aps=None,
        seq_len_dvs=None,
        seq_len_gps=None,
        seq_len_head=None,
        seq_len_time=None,
        redis_config: RedisConfig = None,
        map_location="cpu",
        w_aps_norm=False,
        aps_norm_tensor_path=None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.num_exp = len(exp_idx_list)
        self.exp_idx = exp_idx_list

        self.total_aps = [
            np.sort(os.listdir(data_dir + str(idx) + "/dvs_frames")) for idx in exp_idx_list
        ]
        self.total_dvs = [
            np.sort(os.listdir(data_dir + str(idx) + "/dvs_7ms_3seq"))
            for idx in exp_idx_list
        ]
        self.num_imgs = [len(x) for x in self.total_aps]
        self.raw_pos = [
            np.loadtxt(data_dir + str(idx) + "/position.txt", delimiter=" ")
            for idx in exp_idx_list
        ]
        self.raw_head = [
            np.loadtxt(data_dir + str(idx) + "/direction.txt", delimiter=" ")
            for idx in exp_idx_list
        ]

        self.t_pos = [x[:, 0] for x in self.raw_pos]
        self.t_aps = [[float(x[:-4]) for x in y] for y in self.total_aps]
        self.t_dvs = [[float(x[:-4]) for x in y] for y in self.total_dvs]
        self.data_pos = [idx[:, 0:3] - idx[:, 0:3].min(axis=0) for idx in self.raw_pos]
        self.data_head = [
            idx[:, 0:3] - idx[:, 0:3].min(axis=0) for idx in self.raw_head
        ]
        self.seq_len_aps = seq_len_aps
        self.seq_len_gps = seq_len_gps
        self.seq_len_dvs = seq_len_dvs
        self.seq_len_head = seq_len_head
        self.seq_len_time = seq_len_time
        self.seq_len = max(seq_len_gps, seq_len_aps)
        self.nclass = nclass

        self.lens = len(self.total_aps) - self.seq_len
        self.dvs_data = None
        self.duration = [x[-1] - x[0] for x in self.t_dvs]

        nums = 1e5
        for x in self.total_aps:
            if len(x) < nums:
                nums = len(x)
        for x in self.total_dvs:
            if len(x) < nums:
                nums = len(x)
        for x in self.raw_pos:
            if len(x) < nums:
                nums = len(x)

        self.lens = nums

        self.use_redis = False
        self.redis_config = redis_config
        if redis_config is not None:
            self.use_redis = True

            # connect to redis
            try:
                self.redis_connection_pool = redis.ConnectionPool(
                    host=redis_config.redis_host,
                    port=redis_config.redis_port,
                    db=redis_config.redis_db_num,
                    max_connections=10,
                    socket_timeout=5,
                    socket_keepalive=True,
                    health_check_interval=30,
                )
                print("Connected to Redis.")
            except Exception as e:
                print(f"Failed to connect to Redis: {e}")
                self.use_redis = False
        
        self.map_location = map_location
        
        self.w_aps_norm = w_aps_norm
        self.aps_norm_tensor = None
        if w_aps_norm:
            tensor_names = [f"{i}_{self.seq_len_aps}.pt" for i in exp_idx_list]
            norm_tensor_list = []
            # load all tensors
            for i in tensor_names:
                try:
                    norm_tensor_list.append(torch.load(os.path.join(aps_norm_tensor_path, i), map_location=map_location))
                except Exception as e:
                    print(f"Failed to load tensor {i}: {e}")
                    exit(1)

            # the normalization tensor is the weighted sum of all tensors
            # weighted by the number of images in each experiment
            total_imgs = sum(self.num_imgs)
            norm_tensor_list = [ norm_tensor_list[i] * self.num_imgs[i] / total_imgs for i in range(len(norm_tensor_list)) ]
            self.aps_norm_tensor = torch.stack(norm_tensor_list).sum(dim=0)

    def __del__(self):
        """Close the redis connection if it is open."""
        if self.use_redis:
            self.redis_connection_pool.disconnect()

    def __len__(self):
        return self.lens - self.seq_len * 2

    def __getitem__(self, idx):
        if self.use_redis:
            # establish redis connection
            connection = redis.Redis(
                connection_pool=self.redis_connection_pool
            )
        experiment_selector = np.random.randint(self.num_exp)
        exp_index = self.exp_idx[experiment_selector]
        # print(f"{exp_index}_{idx}")
        idx = max(
            min(idx, self.num_imgs[experiment_selector] - self.seq_len * 2),
            self.seq_len_dvs * 3,
        )
        # frame sequence
        img_seq = []
        load_success = False
        if self.use_redis:
            image_batch_keys = [
                f"{exp_index}_{idx - self.seq_len_aps + i}"
                for i in range(self.seq_len_aps)
            ]
            # get image tensor batch
            try:
                img_seq += [
                    torchvision.transforms.ToPILImage()(
                        torch.load(
                            io.BytesIO(connection.hget("image_tensor",key)),
                            weights_only=True,
                            map_location=self.map_location
                        )
                    )
                    for key in image_batch_keys
                ]
                load_success = True
            except Exception as e:
                print(f"Failed to load image tensor from redis: {e}")
                print("Switching to file loading...")
                load_success = False
                img_seq = []
        
        if not load_success or not self.use_redis:
            for i in range(self.seq_len_aps):
                img_loc = (
                    self.data_dir
                    + str(exp_index)
                    + "/dvs_frames/"
                    + self.total_aps[experiment_selector][
                        idx - self.seq_len_aps + i
                    ]
                )
                img_seq += [Image.open(img_loc).convert("RGB")]
        img_seq_pt = []

        if self.transform:
            for images in img_seq:
                img_seq_pt += [torch.unsqueeze(self.transform(images), 0)]

        img_seq = torch.cat(img_seq_pt, dim=0)
        t_stamps = self.raw_pos[experiment_selector][:, 0]
        t_target = self.t_aps[experiment_selector][idx]
        idx_pos = max(np.searchsorted(t_stamps, t_target), self.seq_len_aps)
        # position sequence
        pos_seq = self.data_pos[experiment_selector][
            idx_pos - self.seq_len_gps : idx_pos, :
        ]
        pos_seq = torch.from_numpy(pos_seq.astype("float32"))
        t_stamps = self.raw_head[experiment_selector][:, 0]
        t_target = self.t_aps[experiment_selector][idx]
        idx_head = max(np.searchsorted(t_stamps, t_target), self.seq_len_aps)
        # direction sequence
        head_seq = self.data_head[experiment_selector][
            idx_head - self.seq_len_gps : idx_pos, 1
        ]
        head_seq = torch.from_numpy(head_seq.astype("float32")).reshape(-1, 1)

        # event(time) sequence
        idx_dvs = (
            np.searchsorted(self.t_dvs[experiment_selector], t_target, sorter=None)
            - 1
        )
        t_stamps = self.t_dvs[experiment_selector][idx_dvs]


        dvs_seq = torch.zeros(self.seq_len_dvs * 3, 2, 130, 173)
        load_success = False
        if self.use_redis:
            dvs_batch_keys = [
                f"{exp_index}_{idx_dvs - self.seq_len_dvs + i + 1}"
                for i in range(self.seq_len_dvs)
            ]
            # get dvs tensor batch
            try:
                preprocessed_dvs_tensors = [
                    torch.load(
                        io.BytesIO(connection.hget("dvs_tensor", key)),
                        weights_only=True,
                        map_location=self.map_location
                    )
                    for key in dvs_batch_keys
                ]
                for i in range(self.seq_len_dvs):
                    dvs_seq[i * 3 : (i + 1) * 3] = preprocessed_dvs_tensors[i]
                load_success = True
            except Exception as e:
                print(f"Failed to load dvs tensor from redis: {e}")
                print("Switching to file loading...")
                load_success = False
                dvs_seq = torch.zeros(self.seq_len_dvs * 3, 2, 130, 173)
        
        if not load_success or not self.use_redis:
            for i in range(self.seq_len_dvs):
                dvs_path = (
                    self.data_dir
                    + str(exp_index)
                    + "/dvs_7ms_3seq/"
                    + self.total_dvs[experiment_selector][
                        idx_dvs - self.seq_len_dvs + i + 1
                    ]
                )
                dvs_buf = torch.load(dvs_path, weights_only=True)
                dvs_buf = dvs_buf.permute([1, 0, 2, 3])
                dvs_seq[i * 3 : (i + 1) * 3] = torch.nn.functional.avg_pool2d(
                    dvs_buf, 2
                )

        ids = int(
            (t_stamps - self.t_dvs[experiment_selector][0])
            / self.duration[experiment_selector]
            * self.nclass
        )
        ids = np.clip(ids, a_min=0, a_max=self.nclass - 1)
        ids = np.array(ids)
        ids = torch.from_numpy(ids).type(torch.long)
        
        if self.w_aps_norm:
            img_seq -= self.aps_norm_tensor
            
        if self.use_redis:
            # release redis connection
            connection.close()
        
        return (img_seq, pos_seq, dvs_seq, head_seq), ids


def Data(
    data_path=None,
    batch_size=None,
    exp_idx=None,
    is_shuffle=True,
    normalize=None,
    nclass=None,
    seq_len_aps=None,
    seq_len_dvs=None,
    seq_len_gps=None,
    seq_len_head=None,
    seq_len_time=None,
    worker_per_data_loader=2,
    redis_config: RedisConfig = None,
    map_location="cpu",
    w_aps_norm=False,
    aps_norm_tensor_path=None,
):
    dataset = SequentialDataset(
        data_dir=data_path,
        exp_idx_list=exp_idx,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((240, 320)),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        ),
        nclass=nclass,
        seq_len_aps=seq_len_aps,
        seq_len_dvs=seq_len_dvs,
        seq_len_gps=seq_len_gps,
        seq_len_head=seq_len_head,
        seq_len_time=seq_len_time,
        redis_config=redis_config,
        map_location=map_location,
        w_aps_norm=w_aps_norm,
        aps_norm_tensor_path=aps_norm_tensor_path,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=is_shuffle,
        drop_last=True,
        batch_size=batch_size,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=worker_per_data_loader,
    )
    return data_loader


def Data_brightness(
    data_path=None,
    batch_size=None,
    exp_idx=None,
    is_shuffle=True,
    normalize=None,
    nclass=None,
    seq_len_aps=None,
    seq_len_dvs=None,
    seq_len_gps=None,
    seq_len_head=None,
    seq_len_time=None,
):
    dataset = SequentialDataset(
        data_dir=data_path,
        exp_idx_list=exp_idx,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((240, 320)),
                torchvision.transforms.ToTensor(),
                normalize,
                torchvision.transforms.ColorJitter(brightness=0.5),
            ]
        ),
        nclass=nclass,
        seq_len_aps=seq_len_aps,
        seq_len_dvs=seq_len_dvs,
        seq_len_gps=seq_len_gps,
        seq_len_head=seq_len_head,
        seq_len_time=seq_len_time,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=is_shuffle,
        drop_last=True,
        batch_size=batch_size,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=8,
    )
    return data_loader


def Data_mask(
    data_path=None,
    batch_size=None,
    exp_idx=None,
    is_shuffle=True,
    normalize=None,
    nclass=None,
    seq_len_aps=None,
    seq_len_dvs=None,
    seq_len_gps=None,
    seq_len_head=None,
    seq_len_time=None,
):
    dataset = SequentialDataset(
        data_dir=data_path,
        exp_idx_list=exp_idx,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((240, 320)),
                torchvision.transforms.ToTensor(),
                normalize,
                torchvision.transforms.RandomCrop(size=128, padding=128),
            ]
        ),
        nclass=nclass,
        seq_len_aps=seq_len_aps,
        seq_len_dvs=seq_len_dvs,
        seq_len_gps=seq_len_gps,
        seq_len_head=seq_len_head,
        seq_len_time=seq_len_time,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=is_shuffle,
        drop_last=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
    )
    return data_loader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


def compute_matches(retrieved_all, ground_truth_info):
    matches = []
    itr = 0
    for retr in retrieved_all:
        if retr == ground_truth_info[itr]:
            matches.append(1)
        else:
            matches.append(0)
        itr = itr + 1
    return matches


def compute_precision_recall(matches, scores_all):
    precision, recall, _ = precision_recall_curve(matches, scores_all)
    return precision, recall


def find_nearest_hash(input_samples, hash_table, hash_table_labels, device="cpu"):
    """Find the nearest hash in the hash table.

    The hash table is a tensor shaped (index, sequence_length, hash_length).
    The input_samples is a tensor shaped (batch, sequence_length, hash_length).
    For each sample (sequence, hash), find the index such that the hash_table[index] has the most similar hash.
    Given a sample (sequence, hash_sample) and a hash matrix (sequence, hash_ref), the similarity is defined as
    the product of the similarity of every hash_sample and hash_ref in the sequence.
    """

    # input_samples: (batch, sequence_length, hash_length)
    # hash_table: (index, sequence_length, hash_length)
    # hash_table_labels: (index)

    # similarity: (batch, index, sequence_length)
    similarity = torch.einsum("bsh,ish->bis", input_samples, hash_table).to(device)

    # similarity: (batch, index)
    similarity = similarity.prod(dim=2)

    # index: (batch)
    index = similarity.argmax(dim=1)

    # labels: (batch)
    labels = hash_table_labels.to(device)[index]

    assert labels.shape == (input_samples.shape[0],)

    return labels


def dice_coefficient_matrix(tensor):
    """Calculate dice coefficient matrix of a group of binary vectors"""
    # tensor: (N, M)
    # N: number of vectors
    # M: length of each vector

    # calculate the sum of each vector
    sum_tensor = tensor.sum(dim=1, keepdim=True)  # (N, 1)

    # calculate the intersection of each pair of vectors
    intersection = tensor @ tensor.t()  # (N, N), 计算每一对向量的交集

    # calculate the dice coefficient matrix
    dice_matrix = 2 * intersection / (sum_tensor + sum_tensor.t())  # (N, N)

    # remove the upper triangular part(including the diagonal) of the matrix
    # set the upper triangular part to -1(as dice coefficient is in [0, 1])
    mask = torch.triu(torch.ones_like(dice_matrix), diagonal=1)
    dice_matrix = dice_matrix * mask - (1 - mask)

    return dice_matrix
