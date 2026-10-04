"""Replay storage of the recipe: compact obs layout, head rewards, timeout-only final-obs copies."""

from __future__ import annotations

import os
import weakref
from collections import deque
from typing import Any

import torch
from rsl_rl_flashsac.storage import MemoryEfficientTorchUniformBuffer
from rsl_rl_flashsac.storage.torch_buffer import Batch
from tensordict import TensorDict


def _host_unregister(ptr: int) -> None:
    try:
        torch.cuda.synchronize()  # no kernel may still address the pages
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:  # noqa: BLE001 - interpreter shutdown
        pass


def _gpu_read_enabled() -> bool:
    return os.environ.get("FDR_HOST_GPU_READ", "1") != "0"


def _host_register_flags() -> int:
    # PORTABLE | MAPPED: the GPU addresses the pages directly (see _mapped_gpu_view)
    return 3 if _gpu_read_enabled() else 0


def _host_empty(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, pin: bool) -> torch.Tensor:
    """Replay rows; pinned rows use cudaHostRegister on an exact-size allocation because torch's pinned
    allocator rounds blocks above 32 GiB up to a power of two (FDR_HOST_REGISTER=0: pin_memory)."""
    if not pin:
        return torch.empty(shape, dtype=dtype, device=device)
    if os.environ.get("FDR_HOST_REGISTER", "1") != "0":
        t = torch.empty(shape, dtype=dtype)
        rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), _host_register_flags())
        if int(rc) == 0:
            # freed-but-registered pages make the next registration of a reused address fail
            weakref.finalize(t, _host_unregister, t.data_ptr())
            return t
        print(f"[buffer] cudaHostRegister failed (rc={rc}); falling back to pin_memory", flush=True)
    return torch.empty(shape, dtype=dtype, device=device, pin_memory=True)


class _CudaBytes:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def _mapped_gpu_view(host: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    """CUDA tensor aliasing page-locked host storage (UVA), so gathers and writes run as device kernels
    over the bus instead of a CPU gather + H2D copy; None if unusable."""
    nbytes = host.numel() * host.element_size()
    if nbytes == 0:
        return None
    try:
        raw = torch.as_tensor(_CudaBytes(host.data_ptr(), nbytes), device=device)
        view = raw.view(host.dtype).view(host.shape)
        # round trip through both address spaces before trusting the alias
        probe = torch.tensor([0x5A, 0xA5, 0x3C], dtype=torch.uint8)
        host.view(-1).view(torch.uint8)[:3] = probe
        ok = bool((raw[:3].cpu() == probe).all())
        raw[:3] = probe.flip(0).to(device)
        torch.cuda.synchronize(device)
        ok = ok and bool((host.view(-1).view(torch.uint8)[:3] == probe.flip(0)).all())
        return view if ok else None
    except Exception as e:  # noqa: BLE001
        print(f"[buffer] mapped GPU view unavailable ({e!r})", flush=True)
        return None


class ReplayObsLayout:
    """Flat env obs <-> stored replay groups (``obs`` + int32 ``aux`` when compacted, else the
    network groups) <-> network groups (``actor``/``critic`` split, or one shared ``obs``)."""

    def __init__(self, full_dim: int, actor_dim: int | None = None, compactor: Any | None = None) -> None:
        self.full_dim = full_dim
        self.actor_dim = actor_dim
        self.compactor = compactor
        if actor_dim is None:
            self.network_groups = {"actor": ["obs"], "critic": ["obs"]}
        else:
            self.network_groups = {"actor": ["actor"], "critic": ["critic"]}

    @property
    def store_groups(self) -> list[str]:
        if self.compactor is not None:
            return ["obs", "aux"]
        return ["obs"] if self.actor_dim is None else ["actor", "critic"]

    @property
    def aux_groups(self) -> tuple[str, ...]:
        return ("aux",) if self.compactor is not None else ()

    def _split(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.actor_dim is None:
            return {"obs": flat}
        return {"actor": flat[:, : self.actor_dim], "critic": flat[:, self.actor_dim :]}

    def to_store(self, flat: torch.Tensor, aux: torch.Tensor | None = None) -> TensorDict:
        """Full flat obs (+ aux) -> stored groups (compacted on the input device)."""
        if self.compactor is None:
            groups = self._split(flat)
        else:
            groups = {"obs": self.compactor.compact(flat), "aux": aux}
        return TensorDict(groups, batch_size=[flat.shape[0]])

    def to_network(self, stored: TensorDict) -> TensorDict:
        """Stored groups -> network groups (rebuilds the compacted columns from aux)."""
        if self.compactor is None:
            return stored
        full = self.compactor.expand(stored["obs"], stored["aux"])
        return TensorDict(self._split(full), batch_size=stored.batch_size)

    def actor_input(self, flat: torch.Tensor) -> TensorDict:
        """Actor groups of a rollout obs: [actor | critic] (train) and actor-only (eval) both work."""
        if self.actor_dim is None:
            return TensorDict({"obs": flat}, batch_size=[flat.shape[0]])
        return TensorDict({"actor": flat[:, : self.actor_dim]}, batch_size=[flat.shape[0]])

    def network_template(self) -> TensorDict:
        return TensorDict(self._split(torch.zeros(1, self.full_dim)), batch_size=[1])

    def store_template(self) -> TensorDict:
        if self.compactor is None:
            return self.network_template()
        groups = {
            "obs": torch.zeros(1, self.compactor.compact_dim),
            "aux": torch.zeros(1, self.compactor.aux_dim, dtype=torch.int32),
        }
        return TensorDict(groups, batch_size=[1])


class ReplayBuffer(MemoryEfficientTorchUniformBuffer):
    """Memory-efficient replay with int32 aux groups, (m, H) head rewards and final-obs copies for
    timeouts only; CPU rows on a CUDA host are page-locked and gathered through a mapped view."""

    _ROW_ATTRS = {"action": "_actions", "reward": "_rewards", "terminated": "_terminateds", "truncated": "_truncateds"}

    def __init__(
        self,
        obs: TensorDict,
        num_actions: int,
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
        device: str,
        obs_storage_dtype: torch.dtype | None = None,
        store_groups: list[str] | None = None,
        aux_groups: tuple[str, ...] = (),
        reward_dim: int = 0,
        action_storage_dtype: torch.dtype | None = None,
    ) -> None:
        """aux_groups keep int32 rows (never cast); reward_dim H > 0 stores (m, H) rewards."""
        self._aux_groups = frozenset(aux_groups)
        self._reward_dim = reward_dim
        self._action_storage_dtype = action_storage_dtype
        self._storage_device = torch.device(device)
        super().__init__(
            obs,
            num_actions,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
            device,
            obs_storage_dtype=obs_storage_dtype,
            store_groups=store_groups,
        )

    def reset(self) -> None:
        m = self._max_length
        self._device = self._storage_device
        self._host: dict[str, torch.Tensor] = {}
        pin = self._device.type == "cpu" and torch.cuda.is_available()

        obs_dtype = self._obs_storage_dtype or torch.float32
        self._observations = {
            key: _host_empty((m, *shape), torch.int32 if key in self._aux_groups else obs_dtype, self._device, pin)
            for key, shape in self._obs_shapes.items()
        }
        action_dtype = self._action_storage_dtype or torch.float32
        self._actions = _host_empty((m, self._num_actions), action_dtype, self._device, pin)
        reward_shape = (m, self._reward_dim) if self._reward_dim else (m,)
        self._rewards = _host_empty(reward_shape, torch.float32, self._device, pin)
        self._terminateds = _host_empty((m,), torch.float32, self._device, pin)
        self._truncateds = _host_empty((m,), torch.float32, self._device, pin)
        self._map_host_storage(pin)

        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0
        self._add_batch_size: int | None = None
        self._timeout_next_observations: dict[int, dict[str, torch.Tensor]] = {}

    def _storage(self) -> dict[str, torch.Tensor]:
        rows = {f"observation/{key}": value for key, value in self._observations.items()}
        rows.update({name: getattr(self, attr) for name, attr in self._ROW_ATTRS.items()})
        return rows

    def _set_storage(self, rows: dict[str, torch.Tensor]) -> None:
        for name, value in rows.items():
            if name.startswith("observation/"):
                self._observations[name.split("/", 1)[1]] = value
            else:
                setattr(self, self._ROW_ATTRS[name], value)

    def _map_host_storage(self, pin: bool) -> None:
        """Swap page-locked host rows for CUDA views of the same pages (FDR_HOST_GPU_READ=0: off);
        add/sample then run as on a CUDA buffer while _host keeps the CPU tensors for save/load."""
        if not (pin and _gpu_read_enabled()):
            return
        device = torch.device("cuda", torch.cuda.current_device())
        host = self._storage()
        views: dict[str, torch.Tensor] = {}
        for name, rows in host.items():
            view = _mapped_gpu_view(rows, device)
            if view is None:
                print(f"[buffer] GPU-direct host read disabled ({name} not mappable); using CPU gather", flush=True)
                return
            views[name] = view
        self._host = host
        self._set_storage(views)
        self._device = device
        print(f"[buffer] host replay rows mapped into {device}: GPU-direct gather/scatter", flush=True)

    def _cpu_rows(self, name: str) -> torch.Tensor:
        """CPU-addressable rows for save/load (pending device writes flushed first)."""
        if self._host:
            torch.cuda.synchronize(self._device)
            return self._host[name]
        return self._storage()[name]

    def _get_n_step_prev_transition(self) -> dict[str, Any]:
        """Upstream n-step fold where (n, H) head rewards share the (n,) episode-end flags."""
        n_step_prev_transition = self._n_step_transitions[0]
        curr_transition = self._n_step_transitions[-1]

        n_step_reward = curr_transition["reward"].clone()
        n_step_terminated = curr_transition["terminated"].clone()
        n_step_truncated = curr_transition["truncated"].clone()
        n_step_next_observation = {key: value.clone() for key, value in curr_transition["next_observation"].items()}

        for n_step_idx in reversed(range(self._n_step - 1)):
            transition = self._n_step_transitions[n_step_idx]
            terminated = transition["terminated"]
            truncated = transition["truncated"]

            done = (terminated.bool() | truncated.bool()).float()
            done_r = done.unsqueeze(-1) if n_step_reward.dim() == 2 else done
            n_step_reward = transition["reward"] + self._gamma * n_step_reward * (1 - done_r)

            done_mask = done.bool()
            n_step_terminated[done_mask] = terminated[done_mask]
            n_step_truncated[done_mask] = truncated[done_mask]
            for key in n_step_next_observation:
                n_step_next_observation[key][done_mask] = transition["next_observation"][key][done_mask]

        n_step_prev_transition["reward"] = n_step_reward
        n_step_prev_transition["terminated"] = n_step_terminated
        n_step_prev_transition["truncated"] = n_step_truncated
        n_step_prev_transition["next_observation"] = n_step_next_observation
        return n_step_prev_transition

    def add(self, transition: Batch) -> None:
        self._n_step_transitions.append(self._copy_transition(transition))
        if len(self._n_step_transitions) < self._n_step:
            return

        prev = self._get_n_step_prev_transition()
        add_batch_size = prev["reward"].shape[0]
        if self._add_batch_size is None:
            self._add_batch_size = add_batch_size
            if self._n_step * add_batch_size >= self._max_length:
                raise ValueError("max_length must be larger than n_step * add_batch_size")
        elif add_batch_size != self._add_batch_size:
            raise ValueError("ReplayBuffer requires a constant add batch size")

        idx_tensor = (torch.arange(add_batch_size, device=self._device) + self._current_idx) % self._max_length
        end_idx = self._current_idx + add_batch_size
        idxs: Any = idx_tensor if end_idx > self._max_length else slice(self._current_idx, end_idx)
        for key, storage in self._observations.items():
            storage[idxs] = prev["observation"][key].to(storage.dtype)
        self._actions[idxs] = prev["action"].to(self._actions.dtype)
        self._rewards[idxs] = prev["reward"]
        self._terminateds[idxs] = prev["terminated"]
        self._truncateds[idxs] = prev["truncated"]

        # Overwritten slots no longer belong to the timeouts they were recorded for
        if self._timeout_next_observations:
            for idx in idx_tensor.cpu().tolist():
                self._timeout_next_observations.pop(idx, None)
        # Terminated rows never bootstrap, so only timeouts keep their final observation
        timeout_mask = prev["truncated"].bool() & ~prev["terminated"].bool()
        if timeout_mask.any():
            final_obs = {
                key: value[timeout_mask].to(self._observations[key].dtype)
                for key, value in prev["next_observation"].items()
            }
            for row, idx in enumerate(idx_tensor[timeout_mask].cpu().tolist()):
                self._timeout_next_observations[idx] = {key: value[row].clone() for key, value in final_obs.items()}

        self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
        self._current_idx = end_idx % self._max_length

    def _obs_batch(self, storage: dict[str, torch.Tensor], idxs: torch.Tensor) -> TensorDict:
        cast = self._obs_storage_dtype is not None
        return TensorDict(
            {
                key: value[idxs].float() if cast and key not in self._aux_groups else value[idxs]
                for key, value in storage.items()
            },
            batch_size=[idxs.shape[0]],
            device=self._device,
        )

    def sample(self, sample_idxs: torch.Tensor | None = None) -> Batch:
        assert self._add_batch_size is not None
        if sample_idxs is None:
            sample_high = self._num_in_buffer - self._n_step * self._add_batch_size
            idxs = torch.randint(0, sample_high, (self._sample_batch_size,), device=self._device)
            if self._num_in_buffer == self._max_length:
                idxs = (idxs + self._current_idx) % self._max_length
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device, dtype=torch.long)

        next_idxs = (idxs + self._n_step * self._add_batch_size) % self._max_length
        next_observation = self._obs_batch(self._observations, next_idxs)
        if self._timeout_next_observations:
            hits = [
                (pos, obs)
                for pos, idx in enumerate(idxs.cpu().tolist())
                if (obs := self._timeout_next_observations.get(idx)) is not None
            ]
            if hits:
                positions = torch.as_tensor([pos for pos, _ in hits], device=self._device)
                for key in self._store_groups:
                    rows = torch.stack([obs[key] for _, obs in hits]).to(self._device)
                    next_observation[key][positions] = rows.to(next_observation[key].dtype)

        return {
            "observation": self._obs_batch(self._observations, idxs),
            "action": self._actions[idxs].float(),
            "reward": self._rewards[idxs],
            "terminated": self._terminateds[idxs],
            "truncated": self._truncateds[idxs],
            "next_observation": next_observation,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n = self._num_in_buffer
        torch.save(
            {
                "observation": {key: self._cpu_rows(f"observation/{key}")[:n] for key in self._store_groups},
                **{name: self._cpu_rows(name)[:n] for name in self._ROW_ATTRS},
                "num_in_buffer": self._num_in_buffer,
                "current_idx": self._current_idx,
                "add_batch_size": self._add_batch_size,
                "timeout_next_observations": self._timeout_next_observations,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load our format or flash_rl's (flat observation + observation_aux) through a CPU mmap:
        a device map_location would double the device memory of a large buffer."""
        dataset = torch.load(path, map_location="cpu", mmap=True)
        if isinstance(dataset["observation"], torch.Tensor):
            dataset = self._from_flash_rl(dataset)
        n = dataset["num_in_buffer"]
        for key in self._store_groups:
            self._cpu_rows(f"observation/{key}")[:n] = dataset["observation"][key]
        for name in self._ROW_ATTRS:
            self._cpu_rows(name)[:n] = dataset[name]
        self._num_in_buffer = n
        self._current_idx = dataset["current_idx"]
        self._add_batch_size = dataset["add_batch_size"]
        self._timeout_next_observations = {
            int(idx): {key: value.to(self._device) for key, value in obs.items()}
            for idx, obs in dataset["timeout_next_observations"].items()
        }
        self._n_step_transitions.clear()

    def _from_flash_rl(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """Split flash_rl's flat observation columns into the float groups (in store order)."""
        float_groups = [key for key in self._store_groups if key not in self._aux_groups]
        widths = [self._obs_shapes[key][-1] for key in float_groups]
        aux_key = next(iter(self._aux_groups), None)
        if aux_key is not None and dataset.get("observation_aux") is None:
            raise ValueError("replay buffer file has no observation_aux but obs compaction is on.")

        def split(flat: torch.Tensor, aux: torch.Tensor | None) -> dict[str, torch.Tensor]:
            groups = dict(zip(float_groups, flat.split(widths, dim=-1)))
            if aux_key is not None:
                groups[aux_key] = aux
            return groups

        timeout_aux = dataset.get("timeout_next_aux") or {}
        return {
            **dataset,
            "observation": split(dataset["observation"], dataset.get("observation_aux")),
            "timeout_next_observations": {
                idx: split(obs, timeout_aux.get(idx))
                for idx, obs in dataset.get("timeout_next_observations", {}).items()
            },
        }
