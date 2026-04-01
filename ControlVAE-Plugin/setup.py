from setuptools import setup, find_packages

setup(
    name="controlvae-plugin",
    version="0.2",
    packages=find_packages(),
    #install_requires=["mlagents==1.2.0.dev0"],
    entry_points={
        "mlagents.trainer_type": [
            "ControlVAE=controlvae_plugin.plugin:get_trainer_and_settings",
            #"ControlVAE-Heading=controlvae_plugin.plugin:get_trainer_and_settings",
        ]
    }
)