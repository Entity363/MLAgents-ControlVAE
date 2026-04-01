# HOW TO CREATE A HIGH LEVEL POLICY [CURRENTLY UNUSABLE]:

### For mlagents you'll need:
#### - An actor
#### - An optimizer
#### - A trainer
#### - Settings

### For ml you'll need:
#### - A neural network
#### - A scheduler
#### - an optimizer
#### - A loss function


## Step 1: Setup
Add a new folder under `high_level`, in this case we'll call it `heading`
Create 3 scripts:
- [heading_actor.py](../ControlVAE-Plugin/controlvae_plugin/high_level/heading/heading_actor.py) : it will be our actor
- [heading_optimizer.py](../ControlVAE-Plugin/controlvae_plugin/high_level/heading/heading_optimizer.py) : it will be our optimizer
- [heading_settings.py](../ControlVAE-Plugin/controlvae_plugin/high_level/heading/heading_settings.py) : it will be the settings
- [heading trainer](../ControlVAE-Plugin/controlvae_plugin/high_level/heading/heading_trainer.py): the trainer

## Step 2: Actor
open `heading_actor`, add the required imports, and create a child class to the ControlVAEActor, like this:
```
class ControlVAEHeadingActor(ControlVAEActor):
    MODEL_EXPORT_VERSION = 3

    def __init__(
        self,
        seed: int,
        observation_spec: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        statistics : torch.Tensor,
        dt : float = 0.02,
        **kargs
    ):
        super().__init__(seed, observation_spec, network_settings, action_spec, statistics, dt, **kargs)
```

add the observations based on what you're feeding from Unity, in this case we are feeding a target velocity and a target world angle, thus we add 2 to the original obs of the base nn:
```

        self.observations_size_vector = observation_spec[0].shape[0]
        #obs is state sz + normalized state obs sz + target sz + normalized target obs sz \
        # + 1 done + 1 prior vs post bool \
        # + 2 for speed + angle
        self.done_sz = 1
        self.priorvpost_sz = 1
        self.heading_sz = 2
        extra = self.done_sz + self.priorvpost_sz + self.heading_sz

        self.state_obs_sz = (self.observations_size_vector - extra) // 2
        self.body_sz = (self.state_obs_sz - 3) // 29  #13 body + 16 body + 3
        self.state_sz = self.body_sz * 13
        self.obs_sz = self.body_sz * 16 + 3

        self.target_sz = self.state_sz
        self.target_obs_sz = self.obs_sz

        #not heading sz because it's used after trig conversion(speed, cos, sin)
        self.processed_heading_sz = 3
        self.task_ob_size = self.obs_sz + self.processed_heading_sz
```

And add an mlp(it will offset the latent space):
```
        self.high_level = ptu.build_mlp(self.task_ob_size, kargs["latent_size"], 3, 256, 'ELU').to(default_device())
```

Add an unpack function for the observations you have(it will be used in the trainer as well):
```
    def unpack_raw(self, raw, state_sz, obs_sz) -> dict:
        """
        raw: [B, obs_dim] coming from Unity (vector obs)
        packed as [state | target | done]
        """
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 1:
                raw = raw.unsqueeze(0)
        elif isinstance(raw, np.ndarray):
            if raw.ndim == 1:
                raw = raw[None, :]

        N = state_sz
        M = obs_sz

        state = raw[:, 0:N] #
        state_obs = raw[:, N:N + M]
        target = raw[:, N + M:N + M + N]
        target_obs = raw[:, N + M + N:N + M + N + M]
        done = raw[:, N + M + N + M:N + M + N + M + 1]
        prior_vs_post = raw[:, N + M + N + M + 1:N + M + N + M + 2]
        heading = raw[:,N + M + N + M + 2:N + M + N + M + 2 + self.heading_sz]

        return {"state": state, "state_obs": state_obs, "target": target, "target_obs": target_obs, \
                 "done": done, "prior_vs_post": prior_vs_post, "heading": heading}
```

Add an action function, and call it in the get_actions_and_stats(in this case the random targets come straight from unity, in other cases you might need to manually create a function to generate them, see [joystick playground](../controlvae-main/PlayGround/joystick_playground.py) ):
```
    @staticmethod
    def target2n_target(state, target):
        if len(state.shape) ==2:
            state = state[None,...]
        if len(target.shape) ==1:
            target = target[None,...]
        if isinstance(target, np.ndarray):
            target = ptu.from_numpy(target)
        if isinstance(state, np.ndarray):
            state = ptu.from_numpy(state)
        facing_direction = get_root_facing(state)
        facing_angle = torch.arctan2(facing_direction[:,2], facing_direction[:,0])
        delta_angle = target[:,1] - facing_angle
        res = torch.cat([target[:,0, None], torch.cos(delta_angle[:,None]), torch.sin(delta_angle[:,None])], dim = -1)
        return res
```
```

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:

        #convert
        obs = inputs[0].to(default_device())
        packed = self.unpack_raw(obs, self.state_sz, self.obs_sz)
        state = packed["state"]
        state_obs = packed["state_obs"]
        target = packed["target"]
        target_obs = packed["target_obs"]
        done = packed["done"]
        prior_vs_post = packed["prior_vs_post"]
        heading = packed["heading"]

        batch = obs.shape[0]
        state = state.reshape(batch, self.body_sz, 13)
        target = target.reshape(batch, self.body_sz, 13)

        """
        if self.count <= 512:
            self._debug_traj.append([x.detach().cpu() for x in inputs])
        if self.count == 512:
            torch.save(self._debug_traj, self._debug_traj_path)
            a = 0
        """
        assert(self.motion_type == ControlVAEType.HEADING)
        
        if self.motion_type == ControlVAEType.HEADING:
            if self.mode == ControlVAEMode.TRAINING:
                latent, mu, _ = self.encoder.encode_prior(state_obs)
                heading_target = self.target2n_target(state, heading)
                task = torch.cat([state_obs, heading_target], dim=1)
                offset = self.high_level(task)
                latent = mu+offset
                action = self.decode(state_obs, latent)

            elif self.mode == ControlVAEMode.INFERENCE:
                latent, mu, _ = self.encoder.encode_prior(state_obs)
                heading_target = self.target2n_target(state, heading)
                task = torch.cat([state_obs, heading_target], dim=1)
                offset = self.high_level(task)
                latent = mu+offset
                action = self.decode(state_obs, latent)

        # convert to AgentAction and run_out
        act = AgentAction(continuous_tensor=action.detach().clone(), discrete_list=None)
        run_out = {"env_action": act.to_action_tuple(clip=True),
                    #"mu_prior": mu_prior.detach().clone(),
                    #"mu_post": mu_post.detach().clone(),
                   }
        
        self.count += 1

        return act, run_out, memories 
```

And finally call it in forward(currently broken, crashes if used):
```

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Forward pass of the Actor for inference. This is required for export to ONNX, and
        the inputs and outputs of this method should not be changed without a respective change
        in the ONNX export code.
        """
        obs = inputs[0].to(default_device())
        packed = self.unpack_raw(obs, self.state_sz, self.obs_sz)
        state = packed["state"]
        state_obs = packed["state_obs"]
        target = packed["target"]
        target_obs = packed["target_obs"]
        done = packed["done"]
        prior_vs_post = packed["prior_vs_post"]
        heading = packed["heading"]

        latent, mu, _ = self.encoder.encode_prior(state_obs, deterministic=True)
        heading_target = self.heading_from_state(state, heading)
        
        task = torch.cat([state_obs, heading_target], dim=1)
        offset = self.high_level(task)
        latent = mu+offset                      # it crashes here!!!!
        action = self.decode(state_obs, latent)

        #Needs to be in [obs, act size]
        batch = obs.shape[0]
        action = action.reshape(batch, self.continuous_act_size)

        #for some reason the version number, memory size and shape from above return 0
        version_tensor = torch.tensor([self.MODEL_EXPORT_VERSION], dtype=torch.float32, device=obs.device)
        memory_tensor = torch.tensor([self.memory_size], dtype=torch.float32, device=obs.device)
        shape_tensor = torch.tensor([self.continuous_act_size], dtype=torch.float32, device=obs.device)

        #exporting
        export_out = [version_tensor, memory_tensor]

        #don't let the names fool you, it's actually prior deterministic in action, and posterior deterministic in action deterministic
        if self.action_spec.continuous_size > 0:
            export_out += [
                action,
                shape_tensor,
                action,
            ]

        #memories drama
        device = obs.device
        if self.memory_size > 0:
            mem_out = memories
            if mem_out is None:
                mem_out = torch.zeros((1, self.memory_size), device=device, dtype=obs.dtype)
            export_out.append(mem_out)

        return tuple(export_out)
```

## Step 3: Settings
Add imports and settings you might need, in this case we just need weights, learning rates, and batch size
```
@attr.s(auto_attribs=True)
class ControlVAEHeadingSetttings(ControlVAESettings):
    heading_lr: float = 0.001
    heading_lr_multiplier: float = 0.99

    weight_dir: float = 1
    weight_speed: float = 0
    weight_fall_down: float = 100
    weight_acs: float = 20

    high_level_batch_sz: int = 512
    high_level_rollout_len: int = 16
```



## Step 4: Optimizer
Add imports

Make it a child of ControlVAEOptimizer:
```
class ControlVAEHeadingOptimizer(ControlVAEOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings, statics = None):
        super().__init__(policy, trainer_settings, statics)

```

Add weights and settings:
```
        self.hyperparameters: ControlVAEHeadingSetttings = cast(
            ControlVAEHeadingSetttings, trainer_settings.hyperparameters
        )

        self.weight = {
            "direction": self.hyperparameters.weight_dir,
            "speed": self.hyperparameters.weight_speed,
            "fall_down": self.hyperparameters.weight_fall_down,
            "acs": self.hyperparameters.weight_acs
        }
        
        self.high_level_optim = RAdam(self.policy.actor.high_level.parameters(), lr=self.hyperparameters.heading_lr)
        lr = lambda epoach: max(self.hyperparameters.heading_lr_multiplier**(epoach), 1e-1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.high_level_optim, lr)
```

Add a module to the base ones, it's required to load the encoder and decoder(and world model) weights:
```
    def get_modules(self):
        modules = {
            "Optimizer:vae_optimizer": self.vae_optimizer,
            "Optimizer:wm_optimizer": self.wm_optimizer,
            "Optimizer:heading_optimizer": self.high_level_optim
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
```

Add an acting function, which is necessary because the trainer function uses the world model and not actual unity observations:
```
    def act_task(self, **obs_info):
        
        n_observation = self.obsinfo2n_obs(obs_info)
        latent, mu, _ = self.policy.actor.encoder.encode_prior(n_observation)    
        n_target = self.policy.actor.target2n_target(obs_info['state'], obs_info['target'])
        
        task = torch.cat([n_observation, n_target], dim=1)
        offset = self.policy.actor.high_level(task)
        if self.policy.actor.dance:
            if n_target[...,2].abs()<0.5:
                latent = latent
            else:
                latent = latent + offset
        else:
            latent = mu+offset
        
        action = self.decode(n_observation, latent)
        return action, {
            'mu': mu,
            'latent': latent,
            'offset': offset
        }
```


Add the train_high_level function, including its loss if you have it:
```
    def train_high_level(self, states, targets):
        rollout_length = states.shape[1]
        cur_state = states[:,0].to(default_device())
        targets = targets.to(default_device())
        cur_observation = state2ob(cur_state)
        n_observation = self.normalize_obs(cur_observation)
        
        loss_name = ['direction', 'speed', 'fall_down', 'acs']
        loss_num = len(self.weight)
        loss = [[] for i in range(loss_num)]

        # speed = np.random.choice(self.env.speed_range, targets[:,0,0].shape)
        # targets[:,:,0] = ptu.from_numpy(speed)[:,None]
        for i in range(rollout_length):
            #synthetic step
            action, info = self.act_task(state = cur_state, target = targets[:,i], n_observation = n_observation)
            cur_state = self.policy.actor.world_model(cur_state, action, n_observation = n_observation)
            cur_observation = state2ob(cur_state)
            n_observation = self.normalize_obs(cur_observation)
            # cal_loss
            loss_tmp = self.get_loss(cur_state, targets[:,i])
            for j, value in enumerate(loss_tmp):
                loss[j].append(value)
            action_loss = torch.mean(info['offset']**2)
            loss[-1].append(action_loss)
        
        #optimizer step
        # weight = [1,1,100,20]
        #weight = [1,0,100,20]
        loss_value = [sum(l)/rollout_length*self.weight[loss_name[i]] for i,l in enumerate(loss)]
        loss_value[0] = loss[0][-1]
        loss = sum(loss_value)
        
        
        self.high_level_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.high_level.parameters(), 1)
        self.high_level_optim.step()
        
        # return
        res = {loss_name[i]: loss_value[i] for i in range(loss_num)}
        return res
```

```
    def get_loss(self, state, target):
        direction = get_root_facing(state)
        delta_angle = torch.atan2(direction[:,2], direction[:,0]) - target[:,1]
        direction_loss = torch.acos(torch.cos(delta_angle).clamp(min=-1+1e-4, max=1-1e-4))/ torch.pi

        root_vel = state[:, 0, 7:10]  # pelvis/root linear velocity
        target_direction = torch.cat(
            [torch.cos(target[:, 1, None]), torch.sin(target[:, 1, None])],
            dim=-1,
        )
        speed_along_target = torch.where(
            target[:, 0] == 0,
            torch.norm(root_vel[:, [0, 2]], dim=-1, p=1),
            torch.einsum('bi,bi->b', root_vel[:, [0, 2]], target_direction),
        )
        speed_loss = torch.abs(speed_along_target - target[:, 0]) / target[:, 0].clamp(min=1)
        
        fall_down_loss = torch.clamp(state[...,0,1], min = 0, max = 0.6)
        fall_down_loss = (0.6 - fall_down_loss)
        fall_down_loss = torch.mean(fall_down_loss)
        
        return direction_loss.mean(), speed_loss.mean(), fall_down_loss

```

## Step 5: Trainer
Add imports and the trainer name(it needs to be different from ControlVAE because of the yaml):

```
logger = get_logger(__name__)

TRAINER_NAME = "ControlVAE-Heading"
```

Setup the inheritance and settings:
```
class ControlVAEHeadingTrainer(ControlVAETrainer):
    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.hyperparameters: ControlVAEHeadingSetttings = cast(
            ControlVAEHeadingSetttings, self.trainer_settings.hyperparameters
        )
```


Add keys for the replay buffer:
```
    @property
    def replay_buffer_keys(self):
        return ['state', 'action', 'heading']
```

Create otpimizer, actor and add the trainer name:
```
    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
    
    def create_optimizer(self) -> TorchOptimizer:
        self.optimizer = ControlVAEHeadingOptimizer(  # type: ignore
            cast(ControlVAEPolicy, self.policy), self.trainer_settings,  # type: ignore
            statics= self.statistics
        )  # type: ignore
        return self.optimizer
    
    def create_actor(self):
        actor = ControlVAEHeadingActor
        return actor
```

And call the training functio for the optimizer, using the keys
```
    @property
    def high_level_data_name_list(self):
        return ['state', 'heading']
        
    def _update_policy(self):
        name_list = self.high_level_data_name_list
        rollout_length = self.hyperparameters.high_level_rollout_len
        batch_sz = self.hyperparameters.high_level_batch_sz
        sub_iter = self.hyperparameters.sub_iter
        data_loader = self.replay_buffer.\
            generate_data_loader(   name_list,
                                    rollout_length,
                                    batch_sz,
                                    sub_iter
                                )
        for batch_dict in data_loader:
            log = self.optimizer.train_high_level(*batch_dict)
        self.optimizer.scheduler.step()
        logger.info(log)

        time3 = time.perf_counter()
        #time to go trough the whole process
        if self._last_iteration_end is not None:
            log['iteration_time'] = time3 - self._last_iteration_end
        self._last_iteration_end = time3

        for key, value in log.items():
            if torch.is_tensor(value):
                value = value.detach().mean().item()
            elif isinstance(value, np.ndarray):
                value = float(np.mean(value))
            else:
                value = float(value)

            self.stats_reporter.add_stat(key, value)
        self.policy.count = 0

        return True
```

## Step 6: Create the config file
Create the config file, and add the hyperparameters, for an example
see [here](../config/high_level/heading/config-heading.yaml)

## Step 7: Registration
Registern the plugin in [plugin.py](../ControlVAE-Plugin/controlvae_plugin/plugin.py)