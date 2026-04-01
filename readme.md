# UNITY CONTROLVAE PLUGIN 0.1.0
Implementation of the [ControlVAE](https://heyuanyao-pku.github.io/Control-VAE/) Algorithm in Unity [ML-Agents](https://github.com/unity-technologies/ml-agents) 
[CURRENTLY INCOMPLETE].

#### ⚠️ note: currently inference is unstable due to mismatch between torch and sentis(the emoji was intentional)

## INSTALLATION(Requires conda):
### Python side:
```
conda env create -f requirements.yml
conda activate control-vae-plugin
pip install mlagents==1.1.0
pip install panda3d
```
#### Install torch:
```
conda install pytorch=2.2.2 pytorch-cuda=11.8 torchvision torchaudio -c pytorch -c nvidia
```
#### If having trouble with running jit(match settings to your cuda version):
```
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### Install Locals:
```
cd ControlVAE-Plugin
pip install -e.

cd ../controlvae-main
pip install -e.

cd ../
pip install -e .
```

### Unity Side:
Download the project file from here: https://drive.google.com/drive/folders/1Qa7TG7pu50j1gAVUXiIVxASu8M_-VUiM?usp=sharing  

Unpack the project file in `ControlVAE-Project/UnityControlVAE1.zip` and build the game in production mode.
Or alternative unpack the build file in `ControlVAE-Project/ControlVAESimple.zip`

## Training:
```
conda activate control-vae-plugin
cd [ROOT FOLDER ON YOUR MACHINE]
python ControlVAE-Plugin/controlvae_plugin/run_training.py
```
Or run `ControlVAE-Plugin/controlvae_plugin/run_training.py` directly from your text editor.

Args can be modified inside the run_training python script, or added to the command line, default ones are:
```
"config/config.yaml", 
"--resume",
"--env=ControlVAE-Project/ControlVAESimple/UnityControlVAE1.exe",
"--no-graphics", 
"--num-envs=5",
```

#### Note: Currently only training works for deambulation, as there's a mismatch between torch and sentis.

## Training Times: 
For an `Intel Core I7-11700K 3,60GHz` and an `NVIDIA RTX 3060 12GB` on 6 parallel threads:
 - iterations per minute: 2.76
 - time to 2.000 iterations: 12 hours
 - time to 20k iterations(full training): 120 hours/5 days

## Inference:
Pretrained models are available here: https://drive.google.com/drive/folders/1m4XGigtKZet646kwean4QsP3KssCDEt8?usp=sharing
### Inside Unity:
open `Maps/SimpleControlVAE` inside of the project, and load the `ControlVAE-5015552.onnx` in the behaviour parameters.
Note that inference behaves much worse than training.

### Alternative:
As a temporary solution, we make use of the training pipeline to act as an inference engine, by letting the code run for prolonged periods of time(as you can see in `config/config-inference.yaml`, the rollout length is set at 2048000, which corresponds to about 40.000 seconds or 12 hours).

To make use of it please run `ControlVAE-Plugin/controlvae_plugin/run_inference.py` and press play in the Unity editor.


## Using custom characters:

### Step 1:
Create a prefab, let's call it `Sim`, and add your custom armature, then add rigid bodies and configurable joints to your desired bones.

Set the joints to slerp drive, and set Position Spring to 10000, Damper to 100 and Maximum Force to 25000.

Make sure to add colliders to the rigid bodies.

Select the feet colliders of the `Sim` and disable all layer overrides except the `Floor` Layer.
If it doesn't exist create one.

### Step 2:
Duplicate the prefab, let's call it `Target`, disable colliders and set all rigid bodies to kinematic.

Import your animations and create an animation controller based on the `Sim` avatar, then add your animations to it.

Add an animator component with the animator and avatar.

### Step 3:
Create a scene, let's call it `Motion` and import both `Sim` and `Target`.

Add a floor with a collider, and add it to the `Floor` Layer.

Add an empty object to the scene, and add a ControlVAEController script to it.

Drag the hips component of `Sim` and `Target` to the `Roots` List.

Check that the Keywords for `Weight`, `End effectors`, `Exclude` and `Head` match the names of the bones of your character based on:
- Weight: is used to multiply the error between `Sim` and `Target` bones, specifically used to prevent fingers from overwhelming the loss control in the agent;
-End effectors: Is used to create a list of end effectors(eg. hands and feet);
- Exclude: is used to exclude certain bones from filling up the full bone list, eg. finger ends and transform components under the hierarchy;
- Head: the head of the character, it's used to define if the character fell and episode needs reset.

Once you are happy with the keywords select the Options of the controller, and do `Fill Maps`.

### Step 4: 
Create a custom `Motion Dataset` by doing `Create->ControlVAE->MotionDataset` in the asset browser. This will be used to compute the Statistics for observations normalization.

Add a `Build Motion Dataset` Script to the empty, add the animations you desire to train on, and set your desired Scale for speed of simulation. 

Press Play and let the animations run.

#### ⚠️ Beware that if you feed too large motions the editor could crash during garbage collection, as Build Motion Dataset iterates trough all the frames it cached during runtime.

After that if you inspect the `Motion Dataset` file you should see a list of frames, obs_mean, obs_std, delta_mean, delta_std.

### Step 5:
Duplicate the scene, let's call it `Training`, select the empty, remove the `Build Motion Dataset` and add a `ControlVAEAgent` script. Add your `MotionDataSet` to it and set `max step` to 2048.

Select the `Behaviour Parameter` and call it your preferred name, set the space size to ([Number of bodies in the Controller] * 29 + 3 ) * 2 + 2, as it's formula is based on:
- State = 13 * num bodies
- Obs = 16 * num bodies + 3
- Target State = 13 * num bodies
- Target Obs = 16 * num bodies + 3
- Done Flag = 1
- Posterior vs Prior = 1

Set `Stacked Vectors` to 1.

Set `Continuus Actions` to 4 * (number of joints) as its structure is based on mlagent's: 
- vector 3 rotation for slerp
- 1 float for max force

Set `Discrete Branches` to 0, `Behaviour Type` to default.

Select the `Decision Requester` and set `Decision Period` to 1, `Decision Step` to 0, and check `Take Actions Between Decisions` to true.

### Step 6: 
Train in python.

### Step 7:
Duplicate the scene, call it `Inference` and set the `Max Step` of the `ControlVAEAgent` to a very large number(2048000 corresponds to 12 hours and should suffice).

Import your trained Onnx file and add it to the `Behaviour Parameters`, then select inference mode.

If the Onnx is very unstable, please consider switching to [Python Inference](#alternative).


## Documentation
See [here](./Docs/HowItsMade.md)

## High Level Policies
See [here](./Docs/HighLevelPolicies.md)
